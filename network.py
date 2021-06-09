"""

"""
import torch.nn as nn

class SimpleModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            *[
                nn.Flatten(),
                nn.Linear(28*28, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            ]

        )
    
    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    
    def __init__(self, num_layers, input_shape, inter_dim=100, num_clasess=10):
        super().__init__()
        if len(input_shape) == 2:
            imsize = input_shape[0]*input_shape[1]
        network = [nn.Linear(imsize, inter_dim), nn.ReLU()]
        for _ in range(num_layers-2):
            network.append(nn.Linear(inter_dim, inter_dim))
            network.append(nn.ReLU())
        network.append(nn.Linear(inter_dim, num_clasess))
        self.net = nn.Sequential(*network)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x)
    