# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)  
# https://www.qinglite.cn/doc/3397647815bec7d27
import torch
import torch.nn as nn

# PE(pos, 2i) = sin(pos/10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
def timestep(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2 #整除
    maxp = torch.tensor(max_period, device = timesteps.device)
    ex = (-torch.log(maxp) * torch.arange(half, device = timesteps.device) / half) #step = 1 start = 0 end = half arange:int64
    freq = torch.exp(ex).to(timesteps.device) # 1*32
    args = timesteps[:, None].float() * freq[None] # 维度扩展 2*32
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim = -1)

    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb

class TimestepBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timestep_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.ts_linear = nn.Linear(timestep_dim, out_channels)

        self.bn1 = nn.BatchNorm2d(out_channels) # 归一化
        self.silu1 = nn.SiLU() # x/(1+e^-x)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu2 = nn.SiLU()

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
        x_conv1 = self.conv1(x) # out_channels *3 * 64 * 64
        t_linear1 = self.ts_linear(timesteps)[:, :, None, None]

        x_add = torch.add(x_conv1, t_linear1)

        x_bn1 = self.bn1(x_add)
        x_silu1 = self.silu1(x_bn1)

        x_conv2 = self.conv2(x_silu1)
        x_bn2 = self.bn2(x_conv2)
        x_silu2 = self.silu2(x_bn2)

        return x_silu2

