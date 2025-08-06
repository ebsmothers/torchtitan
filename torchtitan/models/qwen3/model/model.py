# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torch.nn as nn
from torchtitan.models.qwen3.model.qwen3_attention import Qwen3Attention
from torchtitan.modules.attention import MultiHeadAttention
from torchtitan.modules.feed_forward import FeedForward
from torchtitan.modules.qwen2_rope import Qwen2RotaryPositionalEmbeddings

from torchtitan.modules.transformer import (
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)

# Type alias for attention masks
try:
    from torch.nn.attention.flex_attention import BlockMask

    _MaskType = Union[torch.Tensor, BlockMask]
except ImportError:
    # BlockMask not available, use torch.Tensor only
    _MaskType = torch.Tensor

from .args import Qwen3ModelArgs


def qwen3(model_args: Qwen3ModelArgs) -> nn.Module:
    """
    Builder for Qwen3 model using modular components.
    """
    args = Qwen3ModelArgs()

    # Token embeddings
    tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

    # RoPE
    rope = Qwen2RotaryPositionalEmbeddings(
        dim=args.dim // args.n_heads,
        max_seq_len=args.max_seq_len,
        base=args.rope_base,
    )

    # Create modular attention component
    attention = MultiHeadAttention(
        embed_dim=model_args.dim,
        num_heads=model_args.n_heads,
        num_kv_heads=model_args.n_kv_heads or model_args.n_heads,
        head_dim=model_args.dim // model_args.n_heads,
        wq=nn.Linear(
            model_args.dim,
            model_args.n_heads * (model_args.dim // model_args.n_heads),
            bias=False,
        ),
        wk=nn.Linear(
            model_args.dim,
            (model_args.n_kv_heads or model_args.n_heads)
            * (model_args.dim // model_args.n_heads),
            bias=False,
        ),
        wv=nn.Linear(
            model_args.dim,
            (model_args.n_kv_heads or model_args.n_heads)
            * (model_args.dim // model_args.n_heads),
            bias=False,
        ),
        wo=nn.Linear(
            model_args.n_heads * (model_args.dim // model_args.n_heads),
            model_args.dim,
            bias=False,
        ),
        pos_embeddings=rope,
        max_seq_len=model_args.max_seq_len,
        is_causal=True,
        attn_dropout=0.0,
    )

    # Create modular feed-forward component
    hidden_dim = int(2 * (4 * model_args.dim) / 3)
    if model_args.ffn_dim_multiplier is not None:
        hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
    hidden_dim = model_args.multiple_of * (
        (hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of
    )

    feed_forward = FeedForward(
        w1=nn.Linear(model_args.dim, hidden_dim, bias=False),
        w2=nn.Linear(hidden_dim, model_args.dim, bias=False),
        w3=nn.Linear(model_args.dim, hidden_dim, bias=False),
        activation=nn.SiLU(),
    )

    # Create normalization layers
    attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
    ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

    # Calculate weight initialization std
    if model_args.depth_init:
        weight_init_std = 0.02 / (2 * 1) ** 0.5  # layer_id=0 for template
    else:
        weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    # Create modular transformer layer template (torchtune composable style)
    layer_template = TransformerSelfAttentionLayer(
        attn=attention,
        mlp=feed_forward,
        sa_norm=attention_norm,
        mlp_norm=ffn_norm,
        layer_id=0,  # Template layer
        weight_init_std=weight_init_std,
    )

    # Set up attention function for torchtitan compatibility
    try:
        from torchtitan.models.attention import build_attention

        attention.sdpa = build_attention(
            model_args.use_flex_attn, model_args.attn_mask_type
        )
    except ValueError:
        pass

    # Create final normalization and output layers
    norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
    output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    # Create the modular transformer using torchtune's pattern
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layer_template,  # Single layer to be cloned
        norm=norm,
        output=output,
        max_seq_len=model_args.max_seq_len,
        num_heads=model_args.n_heads,
        head_dim=model_args.dim // model_args.n_heads,
        model_args=model_args,
        num_layers=model_args.n_layers,  # Clone the layer this many times
    )

    return model


def qwen3_32b() -> nn.Module:
    """
    Builder for Qwen3 32B model using modular components.
    """
    args = Qwen3ModelArgs()

    # Token embeddings
    tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

    # RoPE
    rope = Qwen2RotaryPositionalEmbeddings(
        dim=args.dim // args.n_heads,
        max_seq_len=args.max_seq_len,
        base=args.rope_base,
    )

    # Build layers
    layers = nn.ModuleList()
    for _ in range(args.n_layers):
        # Self-attention
        self_attn = Qwen3Attention(
            embed_dim=args.dim,
            num_heads=args.n_heads,
            num_kv_heads=args.n_kv_heads,
            head_dim=args.dim // args.n_heads,
            q_proj=nn.Linear(
                args.dim, args.n_heads * (args.dim // args.n_heads), bias=True
            ),
            k_proj=nn.Linear(
                args.dim, args.n_kv_heads * (args.dim // args.n_heads), bias=True
            ),
            v_proj=nn.Linear(
                args.dim, args.n_kv_heads * (args.dim // args.n_heads), bias=True
            ),
            output_proj=nn.Linear(args.dim, args.dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=args.max_seq_len,
            attn_dropout=0.0,
        )

        # Feed forward
        mlp = FeedForward(
            dim=args.dim,
            hidden_dim=args.intermediate_size,
            linear_class=nn.Linear,
            activation=nn.SiLU(),
        )

        # Layer norms
        sa_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        mlp_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        # Create transformer layer
        from torchtitan.modules.transformer import TransformerSelfAttentionLayer

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=sa_norm,
            mlp_norm=mlp_norm,
        )
        layers.append(layer)

    # Final norm and output projection
    norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
    output = nn.Linear(args.dim, args.vocab_size, bias=False)

    # Build the full model
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        norm=norm,
        output=output,
        max_seq_len=args.max_seq_len,
    )

    return model
