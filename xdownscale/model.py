import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- SRCNN ----------------
class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.model(x)

# ---------------- FSRCNN ----------------
class FSRCNN(nn.Module):
    def __init__(self, d=56, s=12, m=4, upscale_factor=1):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )

        self.mid_parts = [nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )]

        for _ in range(m - 1):
            self.mid_parts.append(nn.Sequential(
                nn.Conv2d(s, s, kernel_size=3, padding=3//2),
                nn.PReLU(s)
            ))

        self.mid_parts = nn.Sequential(*self.mid_parts)

        self.last_part = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),
            nn.ConvTranspose2d(d, 1, kernel_size=9, stride=upscale_factor,
                               padding=9//2, output_padding=upscale_factor - 1)
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_parts(x)
        x = self.last_part(x)
        return x

# ---------------- CARNM ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CARNM(nn.Module):
    def __init__(self, num_channels=1, scale_factor=1, num_residual_groups=2, num_residual_blocks=2, num_channels_rg=64):
        super(CARNM, self).__init__()
        self.scale_factor = scale_factor

        self.entry = ConvBlock(num_channels, num_channels_rg, kernel_size=3, stride=1, padding=1)

        self.residual_groups = nn.ModuleList([
            nn.Sequential(*[
                ConvBlock(num_channels_rg, num_channels_rg, kernel_size=3, stride=1, padding=1)
                for _ in range(num_residual_blocks)
            ])
            for _ in range(num_residual_groups)
        ])

        if scale_factor > 1:
            self.upsample = nn.ConvTranspose2d(num_channels_rg, num_channels_rg, kernel_size=3, stride=scale_factor,
                                               padding=1, output_padding=scale_factor - 1)
        else:
            self.upsample = None

        self.exit = ConvBlock(num_channels_rg, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.entry(x)
        residuals = [rg(x) for rg in self.residual_groups]
        x = sum(residuals)
        if self.upsample is not None:
            x = self.upsample(x)
        return self.exit(x)

class LapSRN(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=1):
        super(LapSRN, self).__init__()

        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(64, in_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)

        if upscale_factor > 1:
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        else:
            self.pixel_shuffle = None

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)

        if self.pixel_shuffle is not None:
            x = self.pixel_shuffle(x)

        return x

class FALSRB(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=32, scale_factor=1):
        super(FALSRB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.residual = self.make_layer(num_features, num_features, 3)

        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        if scale_factor > 1:
            self.upsample = nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, stride=scale_factor, padding=1, output_padding=1)
        else:
            self.upsample = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def make_layer(self, in_channels, out_channels, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.residual(x1)
        x3 = self.relu2(self.conv2(x1 + x2))
        out = self.upsample(x3)
        return out

class ResidualBlock(nn.Module):
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

class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual

class CARN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, upscale_factor=1):
        super(CARN, self).__init__()
        self.entry = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.b1 = ResidualBlock(num_features)
        self.b2 = ResidualBlock(num_features)
        self.b3 = ResidualBlock(num_features)

        self.c1 = nn.Conv2d(num_features * 2, num_features, kernel_size=1)
        self.c2 = nn.Conv2d(num_features * 3, num_features, kernel_size=1)
        self.c3 = nn.Conv2d(num_features * 4, num_features, kernel_size=1)

        if upscale_factor > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor)
            )
        else:
            self.upsample = None

        self.exit = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.entry(x)

        x2 = self.b1(x1)
        x2 = self.c1(torch.cat([x1, x2], dim=1))

        x3 = self.b2(x2)
        x3 = self.c2(torch.cat([x1, x2, x3], dim=1))

        x4 = self.b3(x3)
        x4 = self.c3(torch.cat([x1, x2, x3, x4], dim=1))

        if self.upsample is not None:
            x4 = self.upsample(x4)

        out = self.exit(x4)
        return out

class FALSR_A(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=1):
        super(FALSR_A, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, in_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.pixel_shuffle(self.conv6(x5))
        return x6

class OISRRK2(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=1):
        super(OISRRK2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, in_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        res1 = F.relu(self.conv1(x))
        res2 = F.relu(self.conv2(res1))
        res3 = F.relu(self.conv3(res2))
        res4 = F.relu(self.conv4(res3))
        res5 = self.conv5(res4)
        out = F.pixel_shuffle(res5, self.upscale_factor) + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual

class MDSR(nn.Module):
    def __init__(self, in_channels, upscale_factor, num_blocks):
        super(MDSR, self).__init__()

        self.input_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )

        self.output_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x1 = self.prelu(x1)

        x2 = self.residual_blocks(x1)

        x3 = self.output_conv(x2)
        return x + x3

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual

class SecondOrderChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SecondOrderChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * torch.sigmoid(y)

class SAN(nn.Module):
    def __init__(self, in_channels, upscale_factor, num_blocks, num_heads):
        super(SAN, self).__init__()

        self.input_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )

        self.attention_blocks = nn.Sequential(
            *[SecondOrderChannelAttention(64) for _ in range(num_heads)]
        )

        self.output_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x1 = self.prelu(x1)

        x2 = self.residual_blocks(x1)

        x3 = self.attention_blocks(x2)

        x4 = self.output_conv(x3)
        return x + x4


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResidualChannelAttentionBlock, self).__init__()
        modules_body = []
        for _ in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            modules_body.append(act)
        modules_body.pop() # remove last activation
        self.body = nn.Sequential(*modules_body)
        # channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=bias),
            act,
            nn.Conv2d(n_feat // reduction, n_feat, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res) * res
        res += x
        return res

class RCAN(nn.Module):
    def __init__(self, in_channels, num_blocks, upscale_factor):
        super(RCAN, self).__init__()

        self.input_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualChannelAttentionBlock(64) for _ in range(num_blocks)]
        )

        self.output_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x1 = self.prelu(x1)

        x2 = self.residual_blocks(x1)

        x3 = self.output_conv(x2)
        return x + x3

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- UNet ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1 = nn.Conv2d(channel, channel // reduction, 1, padding=0)
        self.c2 = nn.Conv2d(channel // reduction, channel, 1, padding=0)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.c1(y)
        y = nn.ReLU()(y)
        y = self.c2(y)
        return nn.Sigmoid()(y) * x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.ca = CALayer(out_channels)

    def forward(self, x):
        h0 = self.relu(self.c1(x))
        h1 = self.c2(h0)
        h1 = self.ca(h1)
        return h1

class DLGSANet(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(DLGSANet, self).__init__()

        self.input_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
            Block(64, 64)
        )

        self.output_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.relu(self.input_conv(x))

        x2 = self.blocks(x1)

        x3 = self.output_conv(x2)
        return x + x3

class DPMN(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=1):
        super(DPMN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)

        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(64)

        self.conv10 = nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x

        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.bn3(self.conv4(x)))
        x = self.relu(self.bn4(self.conv5(x)))
        x = self.relu(self.bn5(self.conv6(x)))
        x = self.relu(self.bn6(self.conv7(x)))
        x = self.relu(self.bn7(self.conv8(x)))
        x = self.relu(self.bn8(self.conv9(x)))

        x = self.conv10(x)
        x = torch.add(x, residual)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class SAFMN(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=1):
        super(SAFMN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, in_channels * upscale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = self.conv6(x5)
        out = self.pixel_shuffle(x6)

        return out
