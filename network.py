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
    
class UNet(nn.Module):
    
    def __init__(self,
                 in_channels=1,
                 n_classes=2, 
                 depth=5,
                 wf=6,
                 padding=False,
                 batch_norm=False,
                 do_noise_channel=False,
                 use_identity_residual=False,
                 up_mode='upconv'):
        super().__init__()
        
        self.padding = padding
        self.depth = depth
        self.do_noise_channel = do_noise_channel
        self.use_identity_residual = use_identity_residual
        
        if self.do_noise_channel:
            in_channels += 1
        
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
            
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UNetConvBlock(prev_channels, 2**(wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2**(wf + i)
        
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
            

class UNetConvBlock(nn.Module):
    
    # TODO: implement this
    
    def __init__(self, in_size, out_size, padding, batch_norm):
        super().__init__()
        

class UNetUpBlock(nn.Module):
    
    # TODO: implement this
    
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super().__init__(self)
        
        