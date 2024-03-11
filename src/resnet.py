import torch
import torch.nn as nn
from common import Conv3x3, Conv1x1, AvgPool2x2, Upsample2x
from timestep_embedding import TimestepBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timestep_dim: int, layers: int = 1):
        super().__init__()
        # 先考虑一层的情况 有attention 无downscale和upscale
        self.ts_block = TimestepBlock(in_channels, out_channels, timestep_dim)
        

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
        