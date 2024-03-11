import torch.nn as nn
import torch

class AvgPool2x2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
    
    def forward(self,x):
        return self.pool(x)
    
class Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):
        x_conv = self.conv(x)
        x_bn = self.bn(x_conv)
        x_silu = self.silu(x_bn)
        return x_silu
    
class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):
        x_conv = self.conv(x)
        x_bn = self.bn(x_conv)
        x_silu = self.silu(x_bn)
        return x_silu
    
class Upsample2x(nn.Module):
    # 宽高*2
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # conv transpose 逆卷积
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
    
    def forward(self, x: torch.Tensor):
        x_up = self.up(x)
        x_bn = self.bn(x_up)
        x_silu = self.silu(x_bn)
        return x_silu