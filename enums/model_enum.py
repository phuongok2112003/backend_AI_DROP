from enum import Enum

class ModelType(str, Enum):
    gcn_cnn = "GCN+CNN+DROPOUT+RF"
    gcn = "GCN+DROPOUT+RF"
    sage_cnn = "SAGEConv+CNN+DROPOUT+MLP"
