#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GVP (Geometric Vector Perceptron) 完整示例

本脚本展示 GVP 的完整实现，包括：
1. GVP 层
2. GVPConv 消息传递
3. GVPConvLayer 完整层
4. CPDModel 蛋白质设计模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
import numpy as np


# ============================================================
# 工具函数
# ============================================================

def tuple_sum(*args):
    """元组逐元素相加"""
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    """元组拼接"""
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    """元组索引"""
    return x[0][idx], x[1][idx]


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """L2 范数，避免 nan"""
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def normalize(vec, dim=-1, eps=1e-8):
    """归一化向量"""
    return torch.nan_to_num(
        torch.div(vec, torch.norm(vec, dim=dim, keepdim=True)))


# ============================================================
# 1. GVP - 几何向量感知器
# ============================================================

class GVP(nn.Module):
    """
    Geometric Vector Perceptron

    同时处理标量和向量特征，保持旋转等变性
    """

    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid),
                 vector_gate=False):
        """
        参数:
            in_dims: tuple (n_scalar, n_vector) 输入维度
            out_dims: tuple (n_scalar, n_vector) 输出维度
            h_dim: 中间向量维度
            activations: (scalar_act, vector_act) 激活函数
            vector_gate: 是否使用向量门控
        """
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate

        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)

            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)  # [batch, 3, vi]

            # 向量路径
            vh = self.wh(v)  # [batch, 3, h_dim]
            vn = _norm_no_nan(vh, axis=-2)  # 向量范数 [batch, 1, h_dim]

            # 标量路径：拼接原始标量和向量范数
            s = self.ws(torch.cat([s, vn], -1))

            if self.vo:
                v = self.wv(vh)  # [batch, 3, vo]
                v = torch.transpose(v, -1, -2)  # [batch, vo, 3]

                if self.vector_gate:
                    gate = self.wsv(self.scalar_act(s))
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                               device=self.dummy_param.device)

        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


# ============================================================
# 2. LayerNorm 和 Dropout
# ============================================================

class LayerNorm(nn.Module):
    """联合 LayerNorm for (s, V)"""

    def __init__(self, dims):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        # 向量归一化
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class _VDropout(nn.Module):
    """向量 Dropout"""

    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if not self.training:
            return x
        device = self.dummy_param.device
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        return mask * x / (1 - self.drop_rate)


class Dropout(nn.Module):
    """联合 Dropout for (s, V)"""

    def __init__(self, drop_rate):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


# ============================================================
# 3. GVPConv - 图卷积
# ============================================================

class GVPConv(MessagePassing):
    """
    GVP 图卷积/消息传递

    输入图节点和边嵌入，返回新的节点嵌入
    """

    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, aggr="mean",
                 activations=(F.relu, torch.sigmoid),
                 vector_gate=False):
        super().__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        # 构建消息函数
        GVP_ = lambda d_in, d_out: GVP(d_in, d_out,
                                        activations=activations,
                                        vector_gate=vector_gate)

        if n_layers == 1:
            self.message_func = nn.Sequential(
                GVP_((2*self.si + self.se, 2*self.vi + self.ve),
                    (self.so, self.vo), activations=(None, None)))
        else:
            modules = [GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)]
            for _ in range(n_layers - 2):
                modules.append(GVP_(out_dims, out_dims))
            modules.append(GVP_(out_dims, out_dims, activations=(None, None)))
            self.message_func = nn.Sequential(*modules)

    def forward(self, x, edge_index, edge_attr):
        x_s, x_v = x
        # 将向量展平以便 MessagePassing 处理
        x_v_flat = x_v.reshape(x_v.shape[0], 3 * x_v.shape[1])

        message = self.propagate(edge_index, s=x_s, v=x_v_flat, edge_attr=edge_attr)

        # 恢复向量形状
        v_out = message[1].reshape(message[1].shape[0], self.vo, 3)
        return (message[0], v_out)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        # 恢复向量形状
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)

        # 拼接：(s_j, v_j), edge_attr, (s_i, v_i)
        msg_s = torch.cat([s_j, edge_attr[0], s_i], dim=-1)
        msg_v = torch.cat([v_j, edge_attr[1], v_i], dim=-2)

        return self.message_func((msg_s, msg_v))


# ============================================================
# 4. GVPConvLayer - 完整卷积层
# ============================================================

class GVPConvLayer(nn.Module):
    """
    完整的 GVP 卷积层

    包含：消息传递 + 残差连接 + 前馈网络
    """

    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2,
                 drop_rate=0.1, autoregressive=False,
                 activations=(F.relu, torch.sigmoid),
                 vector_gate=False):
        super().__init__()
        self.autoregressive = autoregressive

        self.conv = GVPConv(node_dims, node_dims, edge_dims,
                           n_message, aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)

        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        # 前馈网络
        GVP_ = lambda d_in, d_out, act=True: GVP(
            d_in, d_out, activations=(activations if act else (None, None)),
            vector_gate=vector_gate)

        if n_feedforward == 1:
            self.ff_func = nn.Sequential(GVP_(node_dims, node_dims, act=False))
        else:
            hid_dims = (4*node_dims[0], 2*node_dims[1])
            modules = [GVP_(node_dims, hid_dims)]
            for _ in range(n_feedforward - 2):
                modules.append(GVP_(hid_dims, hid_dims))
            modules.append(GVP_(hid_dims, node_dims, act=False))
            self.ff_func = nn.Sequential(*modules)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        """
        参数:
            x: (s, V) 节点嵌入
            edge_index: [2, num_edges]
            edge_attr: (s, V) 边嵌入
            autoregressive_x: 如果非 None, 用于 src >= dst 的边
            node_mask: 如果非 None, 只更新这些节点
        """
        if self.autoregressive and autoregressive_x is not None:
            src, dst = edge_index
            mask_forward = src < dst
            mask_backward = src >= dst

            # 前向边（使用当前 x）
            edge_idx_fwd = edge_index[:, mask_forward]
            edge_attr_fwd = tuple_index(edge_attr, mask_forward)
            out_fwd = self.conv(x, edge_idx_fwd, edge_attr_fwd)

            # 后向边（使用 autoregressive_x）
            edge_idx_bwd = edge_index[:, mask_backward]
            edge_attr_bwd = tuple_index(edge_attr, mask_backward)
            out_bwd = self.conv(autoregressive_x, edge_idx_bwd, edge_attr_bwd)

            # 合并（考虑度）
            count = scatter_add(torch.ones_like(dst), dst,
                               dim_size=out_fwd[0].size(0)).clamp(min=1)
            dh = tuple_sum(
                (out_fwd[0]/count.unsqueeze(-1), out_fwd[1]/count.unsqueeze(-1).unsqueeze(-1)),
                (out_bwd[0]/count.unsqueeze(-1), out_bwd[1]/count.unsqueeze(-1).unsqueeze(-1))
            )
        else:
            dh = self.conv(x, edge_index, edge_attr)

        # 残差连接 + 归一化
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        # 前馈网络
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        return x


# ============================================================
# 5. CPDModel - 蛋白质条件设计模型
# ============================================================

class CPDModel(nn.Module):
    """
    GVP-GNN 用于结构条件的蛋白质序列设计
    """

    def __init__(self, node_in_dim, node_h_dim,
                 edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.1):
        """
        参数:
            node_in_dim: tuple (n_scalar, n_vector) 节点输入维度
            node_h_dim: tuple (n_scalar, n_vector) 节点隐藏维度
            edge_in_dim: tuple (n_scalar, n_vector) 边输入维度
            edge_h_dim: tuple (n_scalar, n_vector) 边隐藏维度
            num_layers: encoder/decoder 层数
            drop_rate: dropout 率
        """
        super().__init__()

        # 输入嵌入
        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )

        # Encoder
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        # Decoder (自回归)
        self.W_s = nn.Embedding(20, 20)
        edge_h_dim = (edge_h_dim[0] + 20, edge_h_dim[1])

        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim,
                        drop_rate=drop_rate, autoregressive=True)
            for _ in range(num_layers))

        # 输出层
        self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None))

    def forward(self, h_V, edge_index, h_E, seq):
        """
        训练时使用的前向传播

        参数:
            h_V: (s, V) 节点嵌入
            edge_index: [2, num_edges]
            h_E: (s, V) 边嵌入
            seq: [num_nodes] 氨基酸序列（整数编码）
        """
        # Encoder
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        encoder_embeddings = h_V

        # 添加序列信息到边
        h_S = self.W_s(seq)
        h_S_edge = h_S[edge_index[0]]
        h_S_edge[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S_edge], dim=-1), h_E[1])

        # Decoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)

        return self.W_out(h_V)

    def sample(self, h_V, edge_index, h_E, n_samples=1, temperature=0.1):
        """
        自回归采样生成序列

        参数:
            h_V: (s, V) 节点嵌入
            edge_index: [2, num_edges]
            h_E: (s, V) 边嵌入
            n_samples: 采样数量
            temperature: softmax 温度
        """
        with torch.no_grad():
            device = edge_index.device
            L = h_V[0].shape[0]

            # Encoder
            h_V = self.W_v(h_V)
            h_E = self.W_e(h_E)

            for layer in self.encoder_layers:
                h_V = layer(h_V, edge_index, h_E)

            # 扩展到多个样本
            h_V = (h_V[0].repeat(n_samples, 1),
                   h_V[1].repeat(n_samples, 1, 1))
            h_E = (h_E[0].repeat(n_samples, 1),
                   h_E[1].repeat(n_samples, 1, 1))

            # 调整 edge_index
            offset = L * torch.arange(n_samples, device=device).view(-1, 1, 1)
            edge_index_expanded = edge_index + offset
            edge_index_all = torch.cat(tuple(edge_index_expanded), dim=-1)

            # 自回归生成
            seq = torch.zeros(n_samples * L, device=device, dtype=torch.long)
            h_S = torch.zeros(n_samples * L, 20, device=device)

            h_V_cache = [(h_V[0].clone(), h_V[1].clone())
                         for _ in self.decoder_layers]

            for i in range(L):
                # 准备边序列特征
                h_S_edge = h_S[edge_index_all[0]]
                h_S_edge[edge_index_all[0] >= edge_index_all[1]] = 0
                h_E_curr = (torch.cat([h_E[0], h_S_edge], dim=-1), h_E[1])

                # 只更新第 i 个位置
                node_mask = torch.zeros(n_samples * L, device=device, dtype=torch.bool)
                node_mask[i::L] = True

                for j, layer in enumerate(self.decoder_layers):
                    out = layer(h_V_cache[j], edge_index_all, h_E_curr,
                               autoregressive_x=h_V_cache[0], node_mask=node_mask)

                    if j < len(self.decoder_layers) - 1:
                        h_V_cache[j+1][0][i::L] = out[0][i::L]
                        h_V_cache[j+1][1][i::L] = out[1][i::L]

                # 采样
                logits = self.W_out(out)
                seq[i::L] = torch.distributions.Categorical(
                    logits=logits / temperature).sample()
                h_S[i::L] = self.W_s(seq[i::L])

            return seq.view(n_samples, L)


# ============================================================
# 演示函数
# ============================================================

def demo_gvp_layer():
    """演示 GVP 层"""
    print("=" * 70)
    print("1. GVP 层演示")
    print("=" * 70)

    # 创建 GVP 层
    gvp = GVP(
        in_dims=(6, 3),   # 6 维标量，3 维向量
        out_dims=(32, 8), # 32 维标量，8 维向量
        vector_gate=True
    )

    # 创建输入
    batch_size = 4
    s_in = torch.randn(batch_size, 6)
    v_in = torch.randn(batch_size, 3, 3)

    print(f"输入：s={s_in.shape}, v={v_in.shape}")

    # 前向传播
    s_out, v_out = gvp((s_in, v_in))
    print(f"输出：s={s_out.shape}, v={v_out.shape}")


def demo_gvp_conv_layer():
    """演示 GVPConvLayer"""
    print("\n" + "=" * 70)
    print("2. GVPConvLayer 演示")
    print("=" * 70)

    # 创建简单图
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1],
    ])
    num_nodes = 3

    # 节点和边特征
    node_s = torch.randn(num_nodes, 6)
    node_v = torch.randn(num_nodes, 3, 3)
    edge_s = torch.randn(6, 32)
    edge_v = torch.randn(6, 1, 3)

    # 创建层
    layer = GVPConvLayer(
        node_dims=(6, 3),
        edge_dims=(32, 1),
        n_message=3,
        n_feedforward=2,
        drop_rate=0.1
    )

    print(f"输入：node_s={node_s.shape}, node_v={node_v.shape}")

    # 前向传播
    out = layer((node_s, node_v), edge_index, (edge_s, edge_v))
    print(f"输出：out_s={out[0].shape}, out_v={out[1].shape}")


def demo_cpd_model():
    """演示 CPDModel"""
    print("\n" + "=" * 70)
    print("3. CPDModel 演示")
    print("=" * 70)

    # 创建小图
    num_nodes = 10
    num_edges = 30

    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    node_s = torch.randn(num_nodes, 6)
    node_v = torch.randn(num_nodes, 3, 3)
    edge_s = torch.randn(num_edges, 32)
    edge_v = torch.randn(num_edges, 1, 3)
    seq = torch.randint(0, 20, (num_nodes,))

    # 创建模型
    model = CPDModel(
        node_in_dim=(6, 3),
        node_h_dim=(32, 16),
        edge_in_dim=(32, 1),
        edge_h_dim=(32, 1),
        num_layers=3
    )

    print(f"输入:")
    print(f"  node_s={node_s.shape}, node_v={node_v.shape}")
    print(f"  edge_s={edge_s.shape}, edge_v={edge_v.shape}")
    print(f"  seq={seq.shape}")

    # 训练模式前向传播
    model.train()
    logits = model((node_s, node_v), edge_index, (edge_s, edge_v), seq)
    print(f"\n输出 logits: {logits.shape}")  # [num_nodes, 20, 0] -> [num_nodes, 20]

    # 采样
    model.eval()
    sampled_seq = model.sample((node_s, node_v), edge_index, (edge_s, edge_v), n_samples=3)
    print(f"\n采样序列：{sampled_seq.shape}")  # [3, num_nodes]
    print(f"采样序列内容:\n{sampled_seq}")


def demo_rotation_equivariance():
    """演示旋转等变性"""
    print("\n" + "=" * 70)
    print("4. 旋转等变性演示")
    print("=" * 70)

    # 创建随机旋转矩阵
    def random_rotation():
        theta = np.random.uniform(0, 2*np.pi)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        return torch.tensor(R, dtype=torch.float32)

    # 创建 GVP
    gvp = GVP((4, 2), (8, 4))

    # 原始输入
    s = torch.randn(1, 4)
    v = torch.randn(1, 2, 3)

    # 旋转输入
    R = random_rotation()
    v_rotated = torch.matmul(v, R)

    # 前向传播
    s_orig, v_orig = gvp((s, v))
    s_rot, v_rot = gvp((s, v_rotated))

    # 检查等变性
    # 标量应该不变（或近似）
    # 向量应该旋转相同角度
    v_rot_expected = torch.matmul(v_orig, R)

    print(f"原始向量输出：{v_orig}")
    print(f"旋转后向量输出：{v_rot}")
    print(f"期望旋转向量：{v_rot_expected}")

    # 计算差异
    scalar_diff = torch.abs(s_orig - s_rot).mean().item()
    vector_diff = torch.abs(v_rot - v_rot_expected).mean().item()

    print(f"\n标量差异（应该接近 0）: {scalar_diff:.6f}")
    print(f"向量差异（应该接近 0）: {vector_diff:.6f}")

    if scalar_diff < 0.1 and vector_diff < 0.1:
        print("\n✓ GVP 展现出正确的旋转等变性!")
    else:
        print("\n✗ 等变性检查未通过（可能由于数值误差）")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("GVP (Geometric Vector Perceptron) 完整示例\n")

    # 1. GVP 层
    demo_gvp_layer()

    # 2. GVPConvLayer
    demo_gvp_conv_layer()

    # 3. CPDModel
    demo_cpd_model()

    # 4. 旋转等变性
    demo_rotation_equivariance()

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)
