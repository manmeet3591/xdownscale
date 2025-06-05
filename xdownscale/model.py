import torch
import torch.nn as nn

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
