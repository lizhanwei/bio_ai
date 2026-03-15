#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消息传递机制可视化演示

本脚本详细展示 GNN 中消息传递的每一步过程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# ============================================================
# 1. 基础消息传递类
# ============================================================

class SimpleMessagePassing(MessagePassing):
    """
    简化的消息传递实现
    展示消息传递的三个步骤：Message, Aggregate, Update
    """

    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')  # 聚合方式：add, mean, max

        # 定义消息函数中的权重
        self.W_msg = nn.Linear(in_dim, out_dim)
        self.W_upd = nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index):
        """
        前向传播
        x: [num_nodes, in_dim] - 节点特征
        edge_index: [2, num_edges] - 边索引
        """
        print(f"\n输入特征形状：{x.shape}")
        print(f"边索引形状：{edge_index.shape}")

        # 1. 特征变换（准备发送的消息）
        x_transformed = self.W_msg(x)
        print(f"变换后特征形状：{x_transformed.shape}")

        # 2. propagate 会自动调用 message() 和 aggregate()
        print("\n--- 开始消息传递 ---")
        out = self.propagate(edge_index, x=x_transformed)

        # 3. 更新
        out = self.W_upd(out)
        print(f"输出特征形状：{out.shape}")

        return out

    def message(self, x_j):
        """
        消息函数：为每条边 (i, j) 生成消息
        x_j: 目标节点 j 的邻居 i 的特征

        在 message() 被调用时，可以打印详细信息
        """
        print(f"\n[Message] 消息函数被调用")
        print(f"  输入 x_j 形状：{x_j.shape}")
        print(f"  x_j = {x_j}")
        # 这里直接返回 x_j 作为消息
        return x_j

    def aggregate(self, inputs, index):
        """
        聚合函数：聚合每个节点收到的所有消息
        inputs: 所有消息
        index: 每条消息对应的目标节点索引
        """
        print(f"\n[Aggregate] 聚合函数被调用")
        print(f"  输入消息形状：{inputs.shape}")
        print(f"  目标节点索引：{index}")

        # 调用父类的聚合（使用 self.aggr 指定的方式）
        return super().aggregate(inputs, index)

    def update(self, aggr_out):
        """
        更新函数：用聚合后的消息更新节点表示
        """
        print(f"\n[Update] 更新函数被调用")
        print(f"  聚合结果形状：{aggr_out.shape}")
        return aggr_out


# ============================================================
# 2. 详细的消息传递过程演示
# ============================================================

def demo_detailed_message_passing():
    """
    详细演示消息传递的每一步
    """
    print("=" * 70)
    print("消息传递详细演示")
    print("=" * 70)

    # 创建简单的图:
    #
    #     0 -- 1
    #     |    |
    #     2 -- 3
    #
    # 边：0->1, 0->2, 1->0, 1->3, 2->0, 3->1

    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3],  # 源节点
        [1, 2, 0, 3, 0, 1],  # 目标节点
    ])

    # 节点特征（使用简单的整数以便观察）
    x = torch.tensor([
        [1.0, 0.0, 0.0],  # 节点 0: 红色
        [0.0, 1.0, 0.0],  # 节点 1: 绿色
        [0.0, 0.0, 1.0],  # 节点 2: 蓝色
        [1.0, 1.0, 0.0],  # 节点 3: 黄色
    ], dtype=torch.float32)

    print("\n图结构:")
    print("    0 -- 1")
    print("    |    |")
    print("    2 -- 3")

    print("\n节点特征:")
    for i, feat in enumerate(x):
        print(f"  节点 {i}: {feat.tolist()}")

    print("\n边列表:")
    for src, dst in zip(edge_index[0], edge_index[1]):
        print(f"  {src} -> {dst}")

    # 创建模型
    model = SimpleMessagePassing(in_dim=3, out_dim=3)

    print("\n" + "-" * 70)
    print("前向传播:")
    print("-" * 70)

    # 前向传播
    out = model(x, edge_index)

    print("\n" + "-" * 70)
    print("结果:")
    print("-" * 70)
    print(f"输出:\n{out}")

    # 手动计算验证
    print("\n手动计算验证:")
    print("  节点 0 收到来自：节点 1, 节点 2")
    print(f"  节点 0 输出应近似于：{x[1] + x[2]} = {x[1] + x[2]}")
    print("  节点 1 收到来自：节点 0, 节点 3")
    print(f"  节点 1 输出应近似于：{x[0] + x[3]} = {x[0] + x[3]}")


# ============================================================
# 3. 不同聚合方式的比较
# ============================================================

class AggregationComparison(MessagePassing):
    """比较不同聚合方式"""

    def __init__(self, aggr_type):
        super().__init__(aggr=aggr_type)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


def demo_aggregation_types():
    """演示不同聚合方式"""
    print("\n" + "=" * 70)
    print("不同聚合方式比较")
    print("=" * 70)

    # 创建一个节点有多个邻居的图
    #
    #     1
    #    /|\
    #   0 | 2
    #    \|/
    #     3
    #
    # 节点 3 有三个邻居：0, 1, 2

    edge_index = torch.tensor([
        [0, 1, 2, 3],  # 源
        [3, 3, 3, 0],  # 目标
    ])

    x = torch.tensor([
        [1.0, 0.0],  # 节点 0
        [0.0, 2.0],  # 节点 1
        [3.0, 0.0],  # 节点 2
        [0.0, 1.0],  # 节点 3
    ])

    print("\n图结构：节点 0, 1, 2 都指向节点 3")
    print("节点特征:")
    for i, feat in enumerate(x):
        print(f"  节点 {i}: {feat.tolist()}")

    # 比较不同聚合方式
    aggr_types = ['add', 'mean', 'max']

    for aggr in aggr_types:
        model = AggregationComparison(aggr)
        out = model(x, edge_index)
        print(f"\n{aggr} 聚合:")
        print(f"  节点 3 的输出：{out[3].tolist()}")


# ============================================================
# 4. 带边特征的消息传递
# ============================================================

class EdgeFeatureMessagePassing(MessagePassing):
    """使用边特征的消息传递"""

    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__(aggr='mean')
        self.W = nn.Linear(node_dim + edge_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        """
        x_j: 源节点特征
        edge_attr: 边特征
        """
        # 拼接节点特征和边特征
        combined = torch.cat([x_j, edge_attr], dim=-1)
        print(f"[Message] 拼接后形状：{combined.shape}")
        return self.W(combined)


def demo_edge_features():
    """演示带边特征的消息传递"""
    print("\n" + "=" * 70)
    print("带边特征的消息传递")
    print("=" * 70)

    edge_index = torch.tensor([
        [0, 0, 1, 1],
        [1, 2, 0, 2],
    ])

    x = torch.tensor([
        [1.0, 0.0],  # 节点 0
        [0.0, 1.0],  # 节点 1
        [1.0, 1.0],  # 节点 2
    ])

    # 边特征（例如：边的权重、距离等）
    edge_attr = torch.tensor([
        [0.5],  # 边 0->1 的权重
        [0.3],  # 边 0->2 的权重
        [0.7],  # 边 1->0 的权重
        [0.9],  # 边 1->2 的权重
    ], dtype=torch.float32)

    print("\n节点特征:")
    for i, feat in enumerate(x):
        print(f"  节点 {i}: {feat.tolist()}")

    print("\n边特征:")
    for i, (src, dst, attr) in enumerate(zip(edge_index[0], edge_index[1], edge_attr)):
        print(f"  边 {src}->{dst}: {attr.tolist()}")

    model = EdgeFeatureMessagePassing(node_dim=2, edge_dim=1, out_dim=4)
    out = model(x, edge_index, edge_attr)

    print(f"\n输出:\n{out}")


# ============================================================
# 5. 自回归消息传递（用于序列生成）
# ============================================================

class AutoregressiveMessagePassing(MessagePassing):
    """
    自回归消息传递
    在蛋白质设计中，防止未来信息泄露到过去
    """

    def __init__(self, dim):
        super().__init__(aggr='add')
        self.W = nn.Linear(dim, dim)

    def forward(self, x, edge_index, causal_mask=None):
        """
        causal_mask: 布尔掩码，标记哪些边是"前向"的
        """
        self.causal_mask = causal_mask
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # 如果是前向边（src < dst），返回零（防止信息泄露）
        if self.causal_mask is not None:
            x_j = x_j * (~self.causal_mask).float().unsqueeze(-1)
        return self.W(x_j)


def demo_autoregressive():
    """演示自回归消息传递"""
    print("\n" + "=" * 70)
    print("自回归消息传递（用于序列生成）")
    print("=" * 70)

    # 线性图：0 -> 1 -> 2 -> 3
    edge_index = torch.tensor([
        [0, 1, 2, 1, 2, 3],  # 双向边
        [1, 2, 3, 0, 1, 2],
    ])

    x = torch.randn(4, 8)

    # 因果掩码：src >= dst 的边是"后向"的，可以传递
    #          src < dst 的边是"前向"的，需要阻止
    causal_mask = edge_index[0] < edge_index[1]
    print(f"\n因果掩码：{causal_mask.tolist()}")
    print("True 表示前向边（需要阻止），False 表示后向边（允许）")

    model = AutoregressiveMessagePassing(8)
    out = model(x, edge_index, causal_mask)

    print(f"\n输出形状：{out.shape}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("消息传递机制可视化演示\n")

    # 1. 详细消息传递演示
    demo_detailed_message_passing()

    # 2. 不同聚合方式比较
    demo_aggregation_types()

    # 3. 带边特征的消息传递
    demo_edge_features()

    # 4. 自回归消息传递
    demo_autoregressive()

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)
