import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout_rate),
             nn.Linear(64, 2), nn.Softmax(dim=1) 
        )
    
    def forward(self, x):
        return self.layers(x)
    
 



