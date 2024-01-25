import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, num_mid_layers, mid_dim=32, out_dim=2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Linear(in_dim, mid_dim),
                nn.SiLU(),
            )
        )
        for _ in range(num_mid_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim),
                    nn.SiLU(),
                )
            )
        self.layers.append(nn.Linear(mid_dim, out_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x