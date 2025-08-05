# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn


class FeedForward(nn.Module):
    """This class implements the feed-forward network derived from Llama2.

    This is a modular version adapted for torchtitan that maintains compatibility
    with existing parallelization functions while providing the flexibility
    of torchtune's modular design.

    Args:
        w1 (nn.Module): Gate projection from input dim to hidden dim, fed through activation
            and multiplied by w3.
        w2 (nn.Module): Down projection to output dim.
        w3 (Optional[nn.Module]): Up projection from input dim to hidden dim, multiplied by
            activation(w1). If None, only w1 and w2 are used.
        activation (nn.Module): Activation function to use. Default is nn.SiLU().
    """

    def __init__(
        self,
        *,
        w1: nn.Module,
        w2: nn.Module,
        w3: Optional[nn.Module] = None,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        # Use torchtitan naming convention for compatibility
        self.w1 = w1  # gate_proj
        self.w2 = w2  # down_proj
        self.w3 = w3  # up_proj
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``, where ``in_dim`` is the
                input dimension of both ``w1`` and ``w3``.

        Returns:
            torch.Tensor: output tensor with shape ``(..., out_dim)``, where ``out_dim`` is the
                output dimension of ``w2``.
        """
        h = self.activation(self.w1(x))
        if self.w3 is not None:
            h = h * self.w3(x)
        h = self.w2(h)
        return h
