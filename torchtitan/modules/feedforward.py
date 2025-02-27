from typing import Optional

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        gate_proj: nn.Module,
        down_proj: nn.Module,
        up_proj: Optional[nn.Module] = None,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.w1 = gate_proj
        self.w2 = down_proj
        self.w3 = up_proj
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.w1(x))
        if self.w3 is not None:
            h = h * self.w3(x)
        h = self.w2(h)
        return h
