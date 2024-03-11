import torch
import torch.nn as nn
from common import Conv3x3, Conv1x1, AvgPool2x2, Upsample2x
from timestep_embedding import TimestepBlock
from attention import AttentionBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timestep_dim: int, layers: int = 1):
        super().__init__()
        # 先考虑一层的情况 有attention 无downscale和upscale
        self.ts_block = TimestepBlock(in_channels, out_channels, timestep_dim)
        self.at_block = AttentionBlock(out_channels)
        if in_channels != out_channels: # skip conv
            self.conv_skip = Conv1x1(in_channels, out_channels)
            self.skip = True
        else:
            self.skip = False
        self.conv1 = Conv3x3(out_channels, out_channels)
        self.silu = nn.SiLU()


    def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
        a = self.ts_block(x, timesteps)
        a = self.at_block(a)
        if self.skip:
            a = self.conv_skip(a)
        
        final_x = x + a
        final_x = self.silu(self.conv1(final_x)) #不用finalconv?
        return final_x