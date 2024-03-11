import torch
import torch.nn as nn
from common import Conv3x3, Conv1x1, AvgPool2x2, Upsample2x
from timestep_embedding import TimestepBlock
from attention import AttentionBlock

class ResAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timestep_dim: int, attention:bool = True ,layers: int = 1, downscale: bool = False, upscale: bool = False):
        super().__init__()
        self.skip = in_channels != out_channels
        self.downscale = downscale
        self.upscale = upscale
        self.attention = attention
        # 先考虑一层的情况 有attention 无downscale和upscale
        self.ts_block = TimestepBlock(in_channels, out_channels, timestep_dim)
        if attention:
            self.at_block = AttentionBlock(out_channels)
        if in_channels != out_channels: # skip conv
            self.conv_skip = Conv1x1(in_channels, out_channels)
        self.conv1 = Conv3x3(out_channels, out_channels)
        self.silu = nn.SiLU()

        if downscale:
            self.downscale = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        if upscale:
            self.us = lambda x: nn.functional.interpolate(x, scale_factor=2, mode="nearest") # nearest:最近邻插值 (改变分辨率)


    def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
        a = self.ts_block(x, timesteps)
        if self.attention:
            a = self.at_block(a)
        if self.skip:
            a = self.conv_skip(a)
        
        final_x = x + a
        final_x = self.silu(self.conv1(final_x)) #不用finalconv?

        if self.downscale:
            final_x = self.downscale(final_x)
        if self.upscale:
            final_x = self.upscale(final_x)
        return final_x