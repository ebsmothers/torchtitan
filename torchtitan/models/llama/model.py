# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.norms import build_norm

from torchtitan.modules import (
    FeedForward,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol


def llama3(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500_000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
) -> TransformerDecoder:
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    hidden_dim = (
        intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    )
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )

    layers = nn.ModuleList()
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, hidden_dim, bias=False),
            down_proj=nn.Linear(hidden_dim, embed_dim, bias=False),
            up_proj=nn.Linear(embed_dim, hidden_dim, bias=False),
        )
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )


def llama3_8b() -> TransformerDecoder:
    return llama3(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
    )
