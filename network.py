"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        do_noise_channel=False,
        use_identity_residual=False,
        up_mode='upconv'
    ):
        
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
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x, class_label=None, use_zero_noise=False):
        blocks = []

        do_class_generation = False
        if self.do_noise_channel:
            if do_class_generation:
                x = x * 0 + class_label.float().reshape(-1, 1, 1, 1)

            noise_channel = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3])).to(x)*0 + torch.randn((x.shape[0], 1, 1, 1)).to(x)
            if use_zero_noise:
                noise_channel = noise_channel * 0

            out = torch.cat([x, noise_channel], dim=1)
        else:
            out = x

        for i, down in enumerate(self.down_path):
            out = down(out)
            if i != len(self.down_path) - 1:
                blocks.append(out)
                out = F.max_pool2d(out, 2)

        for i, up in enumerate(self.up_path):
            out = up(out, blocks[-i - 1])

        if self.use_identity_residual:
            res = self.last(out)
            if not do_class_generation:
                res = torch.tanh(res)
                return x + res 
            else:
                return res
        else:
            return self.last(out)
            

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
        