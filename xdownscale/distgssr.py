import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Main model architecture for learning-based super-resolution using angular and spatial information.

    Parameters
    ----------
    angRes : int
        Angular resolution of the light field.
    factor : int
        Upscaling factor for super-resolution.

    Forward Input
    -------------
    x : torch.Tensor
        Input tensor of shape [B, 1, H, W].

    Returns
    -------
    out : torch.Tensor
        Super-resolved output tensor of shape [B, 1, H*factor, W*factor].
    """
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        channels = 64
        n_group = 4
        n_block = 4
        self.angRes = angRes
        self.factor = factor
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False)
        self.disentg = CascadeDisentgGroup(n_group, n_block, angRes, channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * factor ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(factor),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        x = SAI2MacPI(x, self.angRes)
        buffer = self.init_conv(x)
        buffer = self.disentg(buffer)
        buffer_SAI = MacPI2SAI(buffer, self.angRes)
        out = self.upsample(buffer_SAI) + x_upscale
        return out


class CascadeDisentgGroup(nn.Module):
    """
    A cascade of Disentanglement Groups to capture hierarchical spatial-angular dependencies.

    Parameters
    ----------
    n_group : int
        Number of DisentgGroups to use.
    n_block : int
        Number of DisentgBlocks per group.
    angRes : int
        Angular resolution.
    channels : int
        Number of feature channels.

    Forward Input
    -------------
    x : torch.Tensor
        Input tensor of shape [B, C, H, W].

    Returns
    -------
    torch.Tensor
        Output tensor after cascading and residual addition.
    """
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DisentgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)
        return self.conv(buffer) + x


class DisentgGroup(nn.Module):
    """
    A group of Disentanglement Blocks followed by a residual convolution.

    Parameters
    ----------
    n_block : int
        Number of DisentgBlocks in this group.
    angRes : int
        Angular resolution.
    channels : int
        Number of feature channels.

    Forward Input
    -------------
    x : torch.Tensor
        Input feature map.

    Returns
    -------
    torch.Tensor
        Output feature map after processing through blocks and residual convolution.
    """

    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DisentgBlock(angRes, channels))
        self.Block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)
        return self.conv(buffer) + x


class DisentgBlock(nn.Module):
    """
    A disentanglement block that extracts features using spatial, angular, and epipolar pathways.

    Parameters
    ----------
    angRes : int
        Angular resolution.
    channels : int
        Number of input/output feature channels.

    Forward Input
    -------------
    x : torch.Tensor
        Input tensor of shape [B, C, H, W].

    Returns
    -------
    torch.Tensor
        Output tensor after feature fusion and residual addition.
    """

    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels//4, channels//2

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(SpaChannel, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.AngConv = nn.Sequential(
            nn.Conv2d(channels, AngChannel, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(angRes),
        )
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes], stride=[1, angRes], padding=[0, angRes * (angRes - 1)//2], bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(EpiChannel, angRes * EpiChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(angRes),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(SpaChannel + AngChannel + 2 * EpiChannel, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
        )

    def forward(self, x):
        feaSpa = self.SpaConv(x)
        feaAng = self.AngConv(x)
        feaEpiH = self.EPIConv(x)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        buffer = torch.cat((feaSpa, feaAng, feaEpiH, feaEpiV), dim=1)
        buffer = self.fuse(buffer)
        return buffer + x


class PixelShuffle1D(nn.Module):
    """
    Custom 1D pixel shuffle layer to upscale the width of the input tensor.

    Parameters
    ----------
    factor : int
        Upscaling factor for the width dimension.

    Forward Input
    -------------
    x : torch.Tensor
        Input tensor of shape [B, factor*C, H, W].

    Returns
    -------
    torch.Tensor
        Output tensor of shape [B, C, H, W*factor].
    """
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()           # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y


def MacPI2SAI(x, angRes):
    """
    Convert Macro-Pixel Image (MacPI) to Sub-Aperture Image (SAI) format.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor in MacPI format [B, C, H, W].
    angRes : int
        Angular resolution.

    Returns
    -------
    torch.Tensor
        Output tensor in SAI format [B, C, H, W].
    """
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    """
    Convert Sub-Aperture Image (SAI) to Macro-Pixel Image (MacPI) format.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor in SAI format [B, C, H, W].
    angRes : int
        Angular resolution.

    Returns
    -------
    torch.Tensor
        Output tensor in MacPI format [B, C, H, W].
    """
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out
