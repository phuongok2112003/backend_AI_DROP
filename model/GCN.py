import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


def build_edge_index_line(edge_index, max_edges_per_node=100):
  
    # Chuyển edge_index từ [2, num_edges] thành [num_edges, 2]
    edge_index = edge_index.t()
    num_edges = edge_index.shape[0]

    # Bước 1: Lập bản đồ từ đỉnh → các cạnh liên quan
    node_to_edges = dict()
    for eid, (u, v) in enumerate(edge_index.tolist()):
        for node in (u, v):
            if node not in node_to_edges:
                node_to_edges[node] = []
            node_to_edges[node].append(eid)

    # Bước 2: Tạo các cặp cạnh chia sẻ đỉnh (mỗi cặp sẽ là một cạnh trong line graph)
    edge_pairs = set()
    for edges in node_to_edges.values():
        # Giới hạn số lượng cạnh trên mỗi đỉnh để tránh quá tải bộ nhớ
        if len(edges) > max_edges_per_node:
            edges = edges[:max_edges_per_node]
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                e1, e2 = edges[i], edges[j]
                edge_pairs.add((e1, e2))
                edge_pairs.add((e2, e1))  # vì đồ thị vô hướng

    # Bước 3: Tạo edge_index_line mới từ các cặp cạnh
    if edge_pairs:
        src, dst = zip(*edge_pairs)
        edge_index_line = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index_line = torch.empty((2, 0), dtype=torch.long)

    return edge_index_line


# === 2. Định nghĩa mô hình GCN ===
class GraphModel(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, node_output_dim, 
                 edge_input_dim, edge_hidden_dim, edge_output_dim):
        super(GraphModel, self).__init__()

        self.node_gcn1 = GCNConv(node_input_dim, node_hidden_dim)
        self.node_gcn2 = GCNConv(node_hidden_dim, node_output_dim)

        self.edge_gcn1 = GCNConv(edge_input_dim, edge_hidden_dim)
        self.edge_gcn2 = GCNConv(edge_hidden_dim, edge_output_dim)

        self.fc = nn.Linear(node_output_dim + edge_output_dim, node_output_dim)

    def forward(self, data):
        node_features = F.sigmoid(self.node_gcn1(data.x, data.edge_index))
        node_features = self.node_gcn2(node_features, data.edge_index)

        if data.edge_attr is not None:
            edge_index_line = build_edge_index_line(data.edge_index)
            edge_features = F.sigmoid(self.edge_gcn1(data.edge_attr, edge_index_line))
            edge_features = self.edge_gcn2(edge_features, edge_index_line)
        else:
            edge_features = torch.zeros((data.edge_index.shape[1], 16), device=data.x.device)

        node_representation = global_mean_pool(node_features, data.batch)
        edge_representation = global_mean_pool(edge_features, data.batch[data.edge_index[0]])

        graph_representation = torch.cat([node_representation, edge_representation], dim=-1)
        graph_representation = self.fc(graph_representation)

        return graph_representation
