# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Union

import torch
from torch import nn

# Type alias for attention masks
try:
    from torch.nn.attention.flex_attention import BlockMask

    _MaskType = Union[torch.Tensor, BlockMask]
except ImportError:
    # BlockMask not available, use torch.Tensor only
    _MaskType = torch.Tensor

logger = logging.getLogger(__name__)


class Qwen3Attention(nn.Module):
    """
    Qwen3 attention module with QK-norm applied before RoPE.
    This is unusual for most models, but Qwen3 became an exception to the rule.

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            ``num_heads % num_kv_heads == 0``. For standard MHA set ``num_kv_heads == num_heads``,
            for GQA ``num_kv_heads < num_heads``, and for MQA set ``num_kv_heads == 1``.
        head_dim (int): dimension of each head, calculated by ``embed_dim // num_heads``.
        q_proj (nn.Module): projection layer for query.
        k_proj (nn.Module): projection layer for key.
        v_proj (nn.Module): projection layer for value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (Optional[nn.Module]): positional embeddings layer, e.g. RotaryPositionalEmbeddings.
        q_norm (Optional[nn.Module]): normalization layer for query, e.g. RMSNorm.
        k_norm (Optional[nn.Module]): normalization layer for key, must be set if q_norm is.
        max_seq_len (int): maximum sequence length supported by the model.
        is_causal (bool): sets the default mask to causal when no mask is provided
        attn_dropout (float): dropout value passed onto the scaled_dot_product_attention function.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: Optional[nn.Module] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(
                f"attn_dropout ({attn_dropout}) must be between 0.0 and 1.0"
            )

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # Set layers
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings

        # Use flex attention if supported and we are sample packing
        # Import attention function - try to use the same as torchtitan
        try:
            from torchtitan.models.attention import build_attention

            self.sdpa = build_attention(use_flex_attn=False, attn_mask_type="causal")
        except ImportError:
            # Fallback to standard SDPA
            self.sdpa = torch.nn.functional.scaled_dot_product_attention

        # this flag indicates whether to update the kv-cache during forward
        # passes. when disabled, we can have the cache setup but still
        # perform normal forward passes
        self.cache_enabled = False

    def setup_cache(
        self, batch_size: int, dtype: torch.dtype, max_seq_len: int
    ) -> None:
        """Setup key value caches for attention calculation."""

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x s_x x d] for the query
            y (Optional[torch.Tensor]): second input tensor with shape [b x s_y x d], is the input
                for k and v. For self attention, x=y.
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
        """
        # x has shape [b, s_x, d]
        # y has shape [b, s_y, d]
        b, s_x, _ = x.shape
        s_y = y.shape[1] if y is not None else 0

        # q has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # Normalize q
        if self.q_norm is not None:
            q = q.transpose(1, 2)
            q = self.q_norm(q)
            q = q.transpose(1, 2)

        # Apply positional embeddings after q-norm
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

        q = q.transpose(1, 2)

        # Update k and v shape, positional embeddings, and normalization
        # k,v shape [b, s_y, num_kv_heads * head_dim]
        k = self.k_proj(y)
        v = self.v_proj(y)

        # Apply positional embeddings
        # k,v shape: [b, s_y, n_kv, h_d]
        k = k.view(b, s_y, -1, self.head_dim)
        v = v.view(b, s_y, -1, self.head_dim)

        # Normalize k
        if self.k_norm is not None:
            k = k.transpose(1, 2)
            k = self.k_norm(k)
            k = k.transpose(1, 2)

        # Apply positional embeddings after k-norm
        if self.pos_embeddings is not None:
            k = self.pos_embeddings(k, input_pos=input_pos)

        # k,v shape: [b, n_kv, s_y, h_d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # If needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        # k,v shape: [b, n_kv, s, h_d] -> [b, n_h, s, h_d]
        if self.num_heads != self.num_kv_heads:
            expand_shape = (b, self.num_kv_heads, q_per_kv, -1, self.head_dim)
            k = k.unsqueeze(2).expand(expand_shape).flatten(1, 2)
            v = v.unsqueeze(2).expand(expand_shape).flatten(1, 2)

        # Compute attention
        if callable(self.sdpa) and not isinstance(
            self.sdpa, type(torch.nn.functional.scaled_dot_product_attention)
        ):
            # Custom attention function (e.g., from torchtitan)
            output = self.sdpa(xq, xk, xv)
        else:
            # Standard SDPA
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return self.output_proj(output)
