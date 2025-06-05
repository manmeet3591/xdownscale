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
