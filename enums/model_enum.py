from enum import Enum

class ModelType(str, Enum):
    gcn_cnn = "GCN+CNN+SOMTE+RF"
    gcn = "GCN+SOMTE+RF"
    sage_cnn = "SAGEConv+CNN+SOMTE+MLP"
