# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Type alias for attention masks
try:
    from torch.nn.attention.flex_attention import BlockMask

    _MaskType = Union[torch.Tensor, BlockMask]
except ImportError:
    # BlockMask not available, use torch.Tensor only
    _MaskType = torch.Tensor


class MultiHeadAttention(nn.Module):
    """Multi-headed attention layer with support for grouped query
    attention (GQA) introduced in https://arxiv.org/abs/2305.13245v1.

    This is a modular version adapted for torchtitan that maintains compatibility
    with existing parallelization functions while providing the flexibility
    of torchtune's modular design.

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            ``num_heads % num_kv_heads == 0``. For standard MHA set ``num_kv_heads == num_heads``,
            for GQA ``num_kv_heads < num_heads``, and for MQA set ``num_kv_heads == 1``.
        head_dim (int): dimension of each head, calculated by ``embed_dim // num_heads``.
        wq (nn.Module): projection layer for query.
        wk (nn.Module): projection layer for key.
        wv (nn.Module): projection layer for value.
        wo (nn.Module): projection layer for output.
        pos_embeddings (Optional[nn.Module]): positional embeddings layer, e.g. RotaryPositionalEmbeddings.
        q_norm (Optional[nn.Module]): normalization layer for query.
        k_norm (Optional[nn.Module]): normalization layer for key, must be set if q_norm is.
        max_seq_len (int): maximum sequence length supported by the model.
        is_causal (bool): sets the default mask to causal when no mask is provided
        attn_dropout (float): dropout value passed onto the scaled_dot_product_attention function.

    Raises:
        ValueError:
            If ``num_heads % num_kv_heads != 0``, **or**
            if ``embed_dim % num_heads != 0``, **or**
            if ``attn_dropout < 0`` or ``attn_dropout > 1``, **or**
            if q_norm is defined without k_norm or vice versa
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: nn.Module,
        wk: nn.Module,
        wv: nn.Module,
        wo: nn.Module,
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

        # Set layers - using torchtitan naming convention for compatibility
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings

        # Number of repetitions for grouped query attention
        self.n_rep = self.num_heads // self.num_kv_heads

        # Import attention function - try to use the same as torchtitan
        try:
            from torchtitan.models.attention import build_attention

            self.sdpa = build_attention(use_flex_attn=False, attn_mask_type="causal")
        except ImportError:
            # Fallback to standard SDPA
            self.sdpa = torch.nn.functional.scaled_dot_product_attention

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            torch.unsqueeze(x, dim=3)
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass matching the TransformerSelfAttentionLayer calling pattern.

        Args:
            x (torch.Tensor): input tensor with shape [b x s x d] for the query
            y (Optional[torch.Tensor]): second input tensor with shape [b x s x d], is the input
                for k and v. For self attention, x=y. If None, uses x for self-attention.
            mask (Optional[_MaskType]): attention mask
            input_pos (Optional[torch.Tensor]): position ids for the input tokens
            freqs_cis (Optional[torch.Tensor]): precomputed rotary embeddings

        Returns:
            torch.Tensor: output tensor with attention applied

        Notation used for tensor shapes:
            - b: batch size
            - s_x: sequence length for x
            - s_y: sequence length for y
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
        """
        # Use x for both query and key/value if y is not provided (self-attention)
        if y is None:
            y = x

        bs, seqlen, _ = x.shape

        # Project to query, key, value
        xq, xk, xv = self.wq(x), self.wk(y), self.wv(y)

        # Reshape for multi-head attention
        # Use -1 instead of specific head counts to handle tensor parallelism
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Apply rotary embeddings if provided
        if self.pos_embeddings is not None:
            xq = self.pos_embeddings(xq)
            xk = self.pos_embeddings(xk)
        elif freqs_cis is not None:
            # Apply rotary embeddings directly (for compatibility with existing torchtitan code)
            from torchtitan.models.llama3.model.model import apply_rotary_emb

            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Apply normalization if provided
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Repeat k/v heads if n_kv_heads < n_heads (for GQA)
        keys = self.repeat_kv(xk, self.n_rep)
        values = self.repeat_kv(xv, self.n_rep)

        # Transpose for attention computation
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

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

        # Reshape output
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)

        return self.wo(output)
