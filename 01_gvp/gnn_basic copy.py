#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GNN 基础示例

本脚本演示基础的图神经网络操作，包括：
1. 图的表示
2. 简单的消息传递
3. GCN 层实现
4. 节点分类示例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 图的表示
# ============================================================

def demo_graph_representation():
    """演示图的多种表示方法"""
    print("=" * 60)
    print("1. 图的表示")
    print("=" * 60)

    # 考虑以下图:
    #     0 -- 1
    #     |    |
    #     2 -- 3

    # 方法 1: 边列表 (Edge List)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3],  # 源节点
        [1, 2, 0, 3, 0, 1],  # 目标节点
    ])
    print(f"边索引 (edge_index): {edge_index.shape}")
    print(edge_index)

    # 方法 2: 邻接矩阵 (Adjacency Matrix)
    num_nodes = 4
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    print(f"\n邻接矩阵:")
    print(adj_matrix)

    # 方法 3: 邻接列表 (Adjacency List)
    adj_list = {i: [] for i in range(num_nodes)}
    for src, dst in zip(edge_index[0], edge_index[1]):
        adj_list[src.item()].append(dst.item())
    print(f"\n邻接列表:")
    for node, neighbors in adj_list.items():
        print(f"  节点 {node}: {neighbors}")

    # 节点特征
    node_features = torch.randn(num_nodes, 8)  # 每个节点 8 维特征
    print(f"\n节点特征形状：{node_features.shape}")

    return edge_index, node_features


# ============================================================
# 2. 简单的消息传递
# ============================================================

def demo_message_passing():
    """演示消息传递过程"""
    print("\n" + "=" * 60)
    print("2. 消息传递演示")
    print("=" * 60)

    # 使用上面的图
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3],
        [1, 2, 0, 3, 0, 1],
    ])
    num_nodes = 4

    # 初始化节点特征
    h = torch.tensor([
        [1.0, 0.0, 0.0],  # 节点 0
        [0.0, 1.0, 0.0],  # 节点 1
        [0.0, 0.0, 1.0],  # 节点 2
        [1.0, 1.0, 0.0],  # 节点 3
    ])

    print("初始节点特征:")
    for i, feat in enumerate(h):
        print(f"  节点 {i}: {feat.tolist()}")

    # 消息传递：每个节点聚合邻居的平均特征
    print("\n消息传递过程:")
    new_h = torch.zeros_like(h)
    for i in range(num_nodes):
        # 找到节点 i 的邻居
        neighbor_mask = edge_index[1] == i
        neighbors = edge_index[0][neighbor_mask]

        print(f"  节点 {i} 的邻居：{neighbors.tolist()}")

        if len(neighbors) > 0:
            # 平均聚合
            new_h[i] = h[neighbors].mean(dim=0)
        else:
            new_h[i] = h[i]

    print("\n聚合后的节点特征:")
    for i, feat in enumerate(new_h):
        print(f"  节点 {i}: {feat.tolist()}")


# ============================================================
# 3. GCN 层实现
# ============================================================

class GCNLayer(nn.Module):
    """简化版 GCN 层"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        """
        参数:
            x: 节点特征 [num_nodes, in_dim]
            edge_index: 边索引 [2, num_edges]
        """
        num_nodes = x.size(0)

        # 1. 添加自环
        identity = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
        edge_index_with_self = torch.cat([edge_index, identity], dim=1)

        # 2. 计算度
        row, col = edge_index_with_self
        deg = torch.bincount(row, minlength=num_nodes).float()

        # 3. 计算归一化系数
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 4. 特征变换
        h = self.W(x)

        # 5. 消息传递
        out = torch.zeros_like(h)
        for i in range(edge_index_with_self.size(1)):
            src, dst = edge_index_with_self[0, i], edge_index_with_self[1, i]
            out[dst] += norm[i] * h[src]

        return F.relu(out)


def demo_gcn_layer():
    """演示 GCN 层的使用"""
    print("\n" + "=" * 60)
    print("3. GCN 层演示")
    print("=" * 60)

    # 创建图
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3],
        [1, 2, 0, 3, 0, 1],
    ])

    # 创建节点特征
    num_nodes = 4
    in_dim = 8
    x = torch.randn(num_nodes, in_dim)

    # 创建 GCN 层
    gcn = GCNLayer(in_dim, 4)

    print(f"输入特征形状：{x.shape}")

    # 前向传播
    out = gcn(x, edge_index)
    print(f"输出特征形状：{out.shape}")
    print(f"输出特征:\n{out}")


# ============================================================
# 4. 完整的 GNN 模型（用于节点分类）
# ============================================================

class SimpleGNN(nn.Module):
    """简单的 2 层 GNN 用于节点分类"""

    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNLayer(in_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


def demo_node_classification():
    """演示节点分类任务"""
    print("\n" + "=" * 60)
    print("4. 节点分类演示")
    print("=" * 60)

    # 创建数据
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3],
        [1, 2, 0, 3, 0, 1],
    ])

    num_nodes = 4
    x = torch.randn(num_nodes, 8)

    # 标签（假设节点 0,1 是类别 0，节点 2,3 是类别 1）
    labels = torch.tensor([0, 0, 1, 1])

    # 创建模型
    model = SimpleGNN(in_dim=8, hidden_dim=16, num_classes=2)

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    print("训练过程:")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        out = model(x, edge_index)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            pred = out.argmax(dim=-1)
            acc = (pred == labels).float().mean().item()
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.2f}")

    # 最终预测
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=-1)
        print(f"\n最终预测：{pred.tolist()}")
        print(f"真实标签：{labels.tolist()}")


# ============================================================
# 5. 可视化图结构
# ============================================================

def visualize_graph(edge_index, node_features=None, save_path=None):
    """使用 matplotlib 可视化图"""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        # 创建 networkx 图
        G = nx.Graph()

        # 添加边
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            G.add_edge(src, dst)

        # 计算布局
        pos = nx.spring_layout(G, seed=42)

        # 绘制
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=16, font_weight='bold', ax=ax)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图已保存到：{save_path}")
        else:
            plt.show()

    except ImportError:
        print("安装依赖以进行可视化：pip install matplotlib networkx")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("GNN 基础示例\n")

    # 1. 图的表示
    edge_index, node_features = demo_graph_representation()

    # 2. 消息传递
    demo_message_passing()

    # 3. GCN 层
    demo_gcn_layer()

    # 4. 节点分类
    demo_node_classification()

    # 5. 可视化
    print("\n" + "=" * 60)
    print("5. 图可视化")
    print("=" * 60)
    visualize_graph(edge_index, save_path="./graph_viz.png")

    print("\n完成！")
