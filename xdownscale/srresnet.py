import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel (default is 3).

    Input shape:
        (B, in_channels, H, W)

    Output shape:
        (B, out_channels, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual

class SRResNet(nn.Module):
    """
    Super-Resolution Residual Network (SRResNet).

    A deep residual network for image super-resolution using multiple residual blocks.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default is 1).
    out_channels : int
        Number of output channels (default is 1).
    num_features : int
        Number of feature channels (default is 64).
    num_blocks : int
        Number of residual blocks (default is 16).
    upscale_factor : int
        Upscaling factor for super-resolution (default is 1).

    Input shape:
        (B, in_channels, H, W)

    Output shape:
        (B, out_channels, H*upscale_factor, W*upscale_factor)

    Reference:
        Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." CVPR 2017.
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=16, upscale_factor=1):
        super(SRResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features, num_features) for _ in range(num_blocks)]
        )

        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        if upscale_factor > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor)
            )
        else:
            self.upsample = None

        self.conv3 = nn.Conv2d(num_features, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.res_blocks(x1)
        x2 = self.conv2(x2)
        x3 = x1 + x2

        if self.upsample is not None:
            x3 = self.upsample(x3)

        out = self.conv3(x3)
        return out
