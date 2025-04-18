
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, GlobalAttention
import numpy as np
from torch.nn import Linear
import torch

class NodeGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.sigmoid(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class EdgeCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, edge_attr):
        if edge_attr is None:
            return None
        if edge_attr.dim() == 2:
            edge_attr = edge_attr.permute(1, 0).unsqueeze(0)
        edge_attr = F.sigmoid(self.conv1(edge_attr))
        edge_attr = self.conv2(edge_attr)
        return edge_attr.squeeze(0).permute(1, 0)

class GraphModel(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, node_output_dim, 
                 edge_input_dim, edge_output_dim, final_dim):
        super(GraphModel, self).__init__()
        self.node_gcn = NodeGraphSAGE(node_input_dim, node_hidden_dim, node_output_dim)
        self.edge_cnn = EdgeCNN(edge_input_dim, edge_output_dim)
        self.att = GlobalAttention(gate_nn=Linear(node_output_dim, 1))
        graph_feature_dim = node_output_dim + edge_output_dim
        self.fc = nn.Linear(graph_feature_dim, final_dim)

    def forward(self, data):
        node_features = self.node_gcn(data.x, data.edge_index)
        edge_features = self.edge_cnn(data.edge_attr) if data.edge_attr is not None else None
        node_features = self.att(node_features, data.batch)
        if edge_features is not None:
            edge_batch = data.batch[data.edge_index[0]]
            edge_features = global_mean_pool(edge_features, edge_batch)
            graph_features = torch.cat([node_features, edge_features], dim=-1)
        else:
            graph_features = node_features
        
        return  graph_features