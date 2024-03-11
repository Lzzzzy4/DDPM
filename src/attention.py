import torch.nn as nn
import torch

# Scaled Dot-product Attention
# Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
# Q = XW_q K = XW_k V = XW_v 三个权重矩阵可训练
class AttentionBlock(nn.Module):
    def __init__(self, channels : int = 128):
        super().__init__()
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)
        self.linear3 = nn.Linear(channels, channels)
        self.linear4 = nn.Linear(channels, channels)
    
    def forward(self, x:torch.Tensor):
        b,c,h,w = x.shape

        x_T = x.view(b,c,-1).permute(0,2,1) # (b, h*w, c)
        Q = self.linear1(x_T)
        K = self.linear2(x_T)
        V = self.linear3(x_T)
        
        Q = Q.view(b,-1,1,c).permute(0,2,1,3) # (b, 1, h*w, c)
        K = K.view(b,-1,1,c).permute(0,2,1,3)
        V = V.view(b,-1,1,c).permute(0,2,1,3)

        a = nn.functional.scaled_dot_product_attention(Q, K, V)
        a = a.transpose(1, 2).reshape(b, -1, c) # (b, h*w, c)
        a = self.linear4(a)
        a = a.transpose(-1, -2).reshape(b, c, h, w)

        return a + x # residual connection