#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蛋白质图构建演示

本脚本展示如何将蛋白质 3D 结构转换为图结构数据
"""

import torch
import torch.nn.functional as F
import numpy as np
import json


# ============================================================
# 1. 氨基酸映射
# ============================================================

# 氨基酸单字母到整数的映射
AA_TO_INT = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

INT_TO_AA = {v: k for k, v in AA_TO_INT.items()}

# 氨基酸物理化学性质
AA_PROPERTIES = {
    'A': {'pI': 6.00, 'polar': 0, 'hydrophobic': 1},
    'C': {'pI': 5.07, 'polar': 1, 'hydrophobic': 0},
    'D': {'pI': 2.97, 'polar': 1, 'hydrophobic': 0},
    'E': {'pI': 3.22, 'polar': 1, 'hydrophobic': 0},
    'F': {'pI': 5.48, 'polar': 0, 'hydrophobic': 1},
    'G': {'pI': 5.97, 'polar': 0, 'hydrophobic': 0},
    'H': {'pI': 7.59, 'polar': 1, 'hydrophobic': 0},
    'I': {'pI': 6.02, 'polar': 0, 'hydrophobic': 1},
    'K': {'pI': 9.74, 'polar': 1, 'hydrophobic': 0},
    'L': {'pI': 5.98, 'polar': 0, 'hydrophobic': 1},
    'M': {'pI': 5.74, 'polar': 0, 'hydrophobic': 1},
    'N': {'pI': 5.41, 'polar': 1, 'hydrophobic': 0},
    'P': {'pI': 6.30, 'polar': 0, 'hydrophobic': 0},
    'Q': {'pI': 5.65, 'polar': 1, 'hydrophobic': 0},
    'R': {'pI': 10.76, 'polar': 1, 'hydrophobic': 0},
    'S': {'pI': 5.68, 'polar': 1, 'hydrophobic': 0},
    'T': {'pI': 5.60, 'polar': 1, 'hydrophobic': 0},
    'V': {'pI': 5.96, 'polar': 0, 'hydrophobic': 1},
    'W': {'pI': 5.89, 'polar': 0, 'hydrophobic': 1},
    'Y': {'pI': 5.66, 'polar': 1, 'hydrophobic': 0},
}


# ============================================================
# 2. 几何工具函数
# ============================================================

def normalize(vec, dim=-1, eps=1e-8):
    """归一化向量"""
    norm = torch.norm(vec, dim=dim, keepdim=True)
    return vec / (norm + eps)


def rbf_encoding(distances, D_min=0., D_max=20., D_count=16):
    """
    径向基函数编码距离
    将连续距离转换为 D_count 维的编码
    """
    device = distances.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(distances, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def positional_encoding(seq_dist, num_emb=16):
    """
    Transformer 风格的位置编码
    """
    device = seq_dist.device
    frequency = torch.exp(
        torch.arange(0, num_emb, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_emb)
    )
    angles = seq_dist.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)


def dihedral_angles(coords):
    """
    计算蛋白质主链二面角 (phi, psi, omega)

    参数:
        coords: [L, 3, 3] - N, CA, C 坐标

    返回:
        [L, 6] - cos 和 sin 编码的二面角
    """
    # 计算向量
    b = coords[:, :-1, :] - coords[:, 1:, :]  # 键向量

    # 计算法向量
    n1 = normalize(torch.cross(b[0], b[1]), dim=-1)  # 平面 1 的法向量
    n2 = normalize(torch.cross(b[1], b[2]), dim=-1)  # 平面 2 的法向量

    # 二面角
    m1 = torch.cross(n1, normalize(b[1], dim=-1))
    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(m1 * n2, dim=-1)
    angle = torch.atan2(y, x)

    return angle


def calculate_dihedrals(coords, eps=1e-7):
    """
    计算完整的二面角特征（用于 GVP）

    参数:
        coords: [L, 4, 3] - N, CA, C, O 坐标

    返回:
        [L, 6] - cos 和 sin 编码的三个二面角
    """
    L = coords.shape[0]

    # 重新排列用于计算
    X = coords[:, :3, :].reshape(3*L, 3)

    # 键向量
    dX = X[1:] - X[:-1]
    U = normalize(dX, dim=-1)

    # 法向量
    u_0, u_1, u_2 = U[:-2], U[1:-1], U[2:]
    n_1 = normalize(torch.cross(u_0, u_1), dim=-1)
    n_2 = normalize(torch.cross(u_1, u_2), dim=-1)

    # 角度
    cosD = torch.clamp(torch.sum(n_1 * n_2, -1), -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_0 * n_2, -1)) * torch.acos(cosD)

    # 填充
    D = F.pad(D, [1, 2])
    D = D.reshape(-1, 3)

    # sin/cos 编码
    return torch.cat([torch.cos(D), torch.sin(D)], dim=-1)


def calculate_orientations(X_ca):
    """
    计算每个残基的前向和后向单位向量

    参数:
        X_ca: [L, 3] - CA 原子坐标

    返回:
        [L, 2, 3] - 前向和后向向量
    """
    forward = normalize(X_ca[1:] - X_ca[:-1])
    backward = normalize(X_ca[:-1] - X_ca[1:])

    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])

    return torch.stack([forward, backward], dim=1)


def calculate_sidechains(coords):
    """
    计算侧链方向向量

    参数:
        coords: [L, 4, 3] - N, CA, C, O 坐标

    返回:
        [L, 3] - 侧链方向
    """
    N, CA, C = coords[:, 0], coords[:, 1], coords[:, 2]

    C_vec = normalize(C - CA)
    N_vec = normalize(N - CA)

    # 角平分线
    bisector = normalize(C_vec + N_vec)

    # 垂直向量
    perp = normalize(torch.cross(C_vec, N_vec))

    # 侧链方向（近似）
    sidechain = -bisector * np.sqrt(1/3) - perp * np.sqrt(2/3)

    return sidechain


# ============================================================
# 3. 蛋白质图构建
# ============================================================

class ProteinGraphBuilder:
    """蛋白质图构建器"""

    def __init__(self, top_k=30, num_rbf=16, num_pos_emb=16):
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_pos_emb = num_pos_emb

    def build_graph(self, coords, seq, name="protein"):
        """
        构建蛋白质图

        参数:
            coords: [L, 4, 3] - N, CA, C, O 坐标
            seq: 氨基酸序列字符串
            name: 蛋白质名称

        返回:
            dict: 包含图的所有属性和标签
        """
        L = len(seq)

        # 1. 提取 CA 坐标
        X_ca = coords[:, 1, :]  # [L, 3]

        # 2. 构建 kNN 图
        edge_index = self._build_knn_graph(X_ca)

        # 3. 计算节点特征
        node_s, node_v = self._build_node_features(coords, seq, X_ca)

        # 4. 计算边特征
        edge_s, edge_v = self._build_edge_features(X_ca, edge_index)

        # 5. 序列编码
        seq_encoded = torch.tensor([AA_TO_INT[aa] for aa in seq], dtype=torch.long)

        # 6. 掩码（处理缺失坐标）
        mask = torch.isfinite(coords.sum(dim=(1, 2)))

        return {
            'name': name,
            'seq': seq,
            'seq_encoded': seq_encoded,
            'X_ca': X_ca,
            'coords': coords,
            'edge_index': edge_index,
            'node_s': node_s,
            'node_v': node_v,
            'edge_s': edge_s,
            'edge_v': edge_v,
            'mask': mask,
            'num_nodes': L,
            'num_edges': edge_index.shape[1],
        }

    def _build_knn_graph(self, X_ca):
        """构建 kNN 图"""
        # 计算距离矩阵
        diff = X_ca.unsqueeze(0) - X_ca.unsqueeze(1)
        dist_matrix = torch.norm(diff, dim=-1)

        # 获取每个节点的 k 个最近邻
        L = X_ca.shape[0]
        edge_index = []

        for i in range(L):
            distances = dist_matrix[i].clone()
            distances[i] = float('inf')  # 排除自己

            # 获取最近的 k 个邻居
            _, top_indices = torch.topk(distances, min(self.top_k, L-1), largest=False)

            for j in top_indices:
                edge_index.append([i, j.item()])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return edge_index

    def _build_node_features(self, coords, seq, X_ca):
        """构建节点特征"""
        # 二面角特征
        dihedrals = calculate_dihedrals(coords)  # [L, 6]

        # pI 值
        pI_values = torch.tensor([AA_PROPERTIES[aa]['pI'] for aa in seq], dtype=torch.float32).unsqueeze(-1)

        # 标量特征：二面角 + pI
        node_s = torch.cat([dihedrals, pI_values], dim=-1)

        # 向量特征：方向 + 侧链
        orientations = calculate_orientations(X_ca)  # [L, 2, 3]
        sidechains = calculate_sidechains(coords).unsqueeze(1)  # [L, 1, 3]
        node_v = torch.cat([orientations, sidechains], dim=1)

        return node_s, node_v

    def _build_edge_features(self, X_ca, edge_index):
        """构建边特征"""
        # 边向量
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        distances = torch.norm(E_vectors, dim=-1)

        # RBF 编码
        rbf = rbf_encoding(distances, D_count=self.num_rbf)

        # 位置编码
        seq_dist = edge_index[0] - edge_index[1]
        pos_emb = positional_encoding(seq_dist.abs().float(), self.num_pos_emb)

        # 标量边特征
        edge_s = torch.cat([rbf, pos_emb], dim=-1)

        # 向量边特征（归一化的边方向）
        edge_v = normalize(E_vectors).unsqueeze(-2)

        return edge_s, edge_v


# ============================================================
# 4. 示例数据生成
# ============================================================

def generate_fake_protein(length=50):
    """生成假的蛋白质数据用于演示"""
    # 随机序列
    aa_list = list(AA_TO_INT.keys())
    seq = ''.join(np.random.choice(aa_list, length))

    # 螺旋结构的近似坐标
    coords = []
    for i in range(length):
        # 简化螺旋
        t = i * 0.5
        ca_x = 10 * np.cos(t)
        ca_y = 10 * np.sin(t)
        ca_z = i * 1.5

        # 其他原子（相对位置）
        n_x, n_y, n_z = ca_x - 1, ca_y, ca_z - 0.5
        c_x, c_y, c_z = ca_x + 1, ca_y, ca_z + 0.5
        o_x, o_y, o_z = ca_x, ca_y + 1, ca_z + 1

        coords.append([
            [n_x, n_y, n_z],
            [ca_x, ca_y, ca_z],
            [c_x, c_y, c_z],
            [o_x, o_y, o_z],
        ])

    coords = torch.tensor(coords, dtype=torch.float32)
    return coords, seq


# ============================================================
# 5. 可视化
# ============================================================

def visualize_protein_graph(graph_dict, save_path=None):
    """可视化蛋白质图"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        X_ca = graph_dict['X_ca'].numpy()
        edge_index = graph_dict['edge_index'].numpy()

        fig = plt.figure(figsize=(12, 6))

        # 3D 结构
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(X_ca[:, 0], X_ca[:, 1], X_ca[:, 2], c='blue', s=50, label='CA')

        # 绘制边
        for src, dst in edge_index.T:
            ax1.plot(
                [X_ca[src, 0], X_ca[dst, 0]],
                [X_ca[src, 1], X_ca[dst, 1]],
                [X_ca[src, 2], X_ca[dst, 2]],
                'gray', alpha=0.3, linewidth=0.5
            )

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Structure')

        # 节点特征热图
        ax2 = fig.add_subplot(122)
        node_s = graph_dict['node_s'].numpy()
        im = ax2.imshow(node_s.T, cmap='coolwarm', aspect='auto')
        ax2.set_xlabel('Residue Index')
        ax2.set_ylabel('Feature')
        ax2.set_title('Node Features (Scalar)')
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图已保存到：{save_path}")
        else:
            plt.show()

    except ImportError:
        print("安装依赖：pip install matplotlib")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("蛋白质图构建演示\n")
    print("=" * 70)

    # 1. 生成示例数据
    print("1. 生成示例蛋白质数据...")
    coords, seq = generate_fake_protein(length=30)
    print(f"   序列长度：{len(seq)}")
    print(f"   序列：{seq[:20]}...")
    print(f"   坐标形状：{coords.shape}")

    # 2. 构建图
    print("\n2. 构建蛋白质图...")
    builder = ProteinGraphBuilder(top_k=5)
    graph = builder.build_graph(coords, seq, name="demo_protein")

    print(f"\n   图属性:")
    print(f"   - 节点数：{graph['num_nodes']}")
    print(f"   - 边数：{graph['num_edges']}")
    print(f"   - 节点标量特征形状：{graph['node_s'].shape}")
    print(f"   - 节点向量特征形状：{graph['node_v'].shape}")
    print(f"   - 边标量特征形状：{graph['edge_s'].shape}")
    print(f"   - 边向量特征形状：{graph['edge_v'].shape}")

    # 3. 特征分析
    print("\n3. 特征分析:")
    print(f"   - 节点标量特征范围：[{graph['node_s'].min():.3f}, {graph['node_s'].max():.3f}]")
    print(f"   - 边标量特征范围：[{graph['edge_s'].min():.3f}, {graph['edge_s'].max():.3f}]")

    # 4. 可视化
    print("\n4. 可视化:")
    visualize_protein_graph(graph, save_path="./protein_graph_viz.png")

    print("\n" + "=" * 70)
    print("演示完成!")
