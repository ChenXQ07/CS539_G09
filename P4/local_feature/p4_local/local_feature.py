"""Local branch matching the supplied RSSS GlobalLocalAttention local path."""
import torch
from torch import nn

class LocalFeatureBranch(nn.Module):
    """Conv1x1-BN plus Conv3x3-BN; input/output [B,C,H,W].

    Input is an already prepared/normalized decoder-block feature. Global
    attention, common projection, residual, MLP and heads are shared-decoder work.
    """
    def __init__(self, channels=64):
        super().__init__()
        if not isinstance(channels,int) or isinstance(channels,bool) or channels<=0:
            raise ValueError("channels must be a positive integer")
        self.channels=channels
        self.local1=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels))
        self.local2=nn.Sequential(nn.Conv2d(channels,channels,1,bias=False),nn.BatchNorm2d(channels))

    def forward(self,x):
        if not isinstance(x,torch.Tensor):
            raise TypeError("expected a torch.Tensor")
        if x.ndim!=4 or x.shape[1]!=self.channels or any(d<=0 for d in x.shape):
            raise ValueError(f"expected nonempty [B,{self.channels},H,W]")
        if not x.is_floating_point():
            raise TypeError("expected floating-point features")
        return self.local2(x)+self.local1(x)
