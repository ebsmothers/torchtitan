# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Modular Llama model implementation using torchtune-style modular components.

This demonstrates how to recreate torchtune's modular model definitions in torchtitan
while maintaining compatibility with existing parallelization functions.
"""

from typing import List, Optional, Union

import torch
from torch import nn

from torchtitan.models.attention import build_attention, init_attention_mask
from torchtitan.models.llama3.model.model import precompute_freqs_cis
from torchtitan.modules import FeedForward, MultiHeadAttention
from torchtitan.protocols.train_spec import ModelProtocol

from .args import TransformerModelArgs


class ModularTransformerBlock(nn.Module):
    """
    Modular TransformerBlock following torchtune's composable design exactly.

    This is a standalone class that takes individual components (attn, mlp, sa_norm, mlp_norm)
    as parameters, just like torchtune's TransformerSelfAttentionLayer, making it fully composable.
    No inheritance - follows torchtune's minimal inheritance philosophy.
    """

    def __init__(
        self,
        *,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        sa_norm: nn.Module,
        mlp_norm: nn.Module,
        layer_id: int = 0,
        weight_init_std: float = 0.02,
    ):
        super().__init__()

        # Store components using original naming for parallelization compatibility
        # This ensures apply_tp() can find the modules and create DTensors correctly
        self.attention = attn  # Changed from self.attn for TP compatibility
        self.feed_forward = mlp  # Changed from self.mlp for TP compatibility
        self.attention_norm = sa_norm  # Changed from self.sa_norm for TP compatibility
        self.ffn_norm = mlp_norm  # Changed from self.mlp_norm for TP compatibility

        # Store attributes for initialization
        self.layer_id = layer_id
        self.weight_init_std = weight_init_std

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass implementing the transformer block logic directly.

        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, dim]
            freqs_cis (torch.Tensor): Precomputed frequency tensor for RoPE

        Returns:
            torch.Tensor: Output tensor [batch, seq_len, dim]
        """
        # Self-attention with residual connection
        # Normalize input
        attn_input = self.attention_norm(x)

        # Apply attention (match TransformerSelfAttentionLayer calling pattern)
        attn_output = self.attention(attn_input, attn_input, freqs_cis=freqs_cis)

        # Residual connection
        h = x + attn_output

        # Feed-forward with residual connection
        # Normalize input
        mlp_input = self.ffn_norm(h)

        # Apply MLP
        mlp_output = self.feed_forward(mlp_input)

        # Residual connection
        output = h + mlp_output

        return output

    def init_weights(self):
        """Initialize weights for the modular components."""
        # Initialize normalization layers
        for norm in (self.attention_norm, self.ffn_norm):
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

        # Initialize attention weights
        for linear in (self.attention.wq, self.attention.wk, self.attention.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(
            self.attention.wo.weight, mean=0.0, std=self.weight_init_std
        )

        # Initialize feed-forward weights
        nn.init.trunc_normal_(self.feed_forward.w1.weight, mean=0.0, std=0.02)
        for linear in (self.feed_forward.w2, self.feed_forward.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=self.weight_init_std)


class ModularTransformer(nn.Module, ModelProtocol):
    """
    Modular Transformer following torchtune's pattern exactly.

    This is a standalone implementation that takes components as parameters,
    just like torchtune's TransformerDecoder, while maintaining compatibility
    with torchtitan's parallelization functions.
    """

    def __init__(
        self,
        *,
        tok_embeddings: nn.Embedding,
        layers: Union[nn.Module, List[nn.Module], nn.ModuleList, nn.ModuleDict],
        norm: nn.Module,
        output: nn.Linear,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        model_args: TransformerModelArgs,
        num_layers: Optional[int] = None,
    ):
        super().__init__()

        # Handle different layer input types (following torchtune's pattern)
        if isinstance(layers, nn.ModuleDict):
            # Keep as ModuleDict for torchtitan compatibility
            pass
        elif isinstance(layers, nn.ModuleList):
            # Convert to ModuleDict for torchtitan compatibility
            layers = nn.ModuleDict({str(i): layer for i, layer in enumerate(layers)})
        elif isinstance(layers, list):
            # Convert to ModuleDict for torchtitan compatibility
            layers = nn.ModuleDict({str(i): layer for i, layer in enumerate(layers)})
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            if num_layers is None:
                raise AssertionError("num_layers is not defined, layers must be a list")
            # Clone the single layer num_layers times
            from torchtitan.modules.transformer import _get_clones

            layer_list = _get_clones(layers, num_layers)
            layers = nn.ModuleDict(
                {str(i): layer for i, layer in enumerate(layer_list)}
            )

        # Store components (torchtune pattern)
        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Store model args and other attributes for torchtitan compatibility
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = len(self.layers)

        # Register freqs_cis buffer
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # Initialize weights
        self.init_weights()

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def init_weights(self, buffer_device: torch.device | None = None):
        """Initialize weights for the modular model."""
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()

        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

        for layer in self.layers.values():
            if layer is not None and hasattr(layer, "init_weights"):
                layer.init_weights()

        if self.norm is not None:
            self.norm.reset_parameters()

        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        eos_id: int | None = None,
        input_batch: torch.Tensor | None = None,
    ):
        """
        Forward pass following torchtune's pattern with torchtitan compatibility.

        Args:
            tokens (torch.Tensor): Input token indices
            eos_id (int | None): End of sequence token id
            input_batch (torch.Tensor | None): Input batch for document masking

        Returns:
            torch.Tensor: Output logits
        """
        # Handle flex attention initialization if needed
        if self.model_args.use_flex_attn:
            init_attention_mask(
                input_batch if input_batch is not None else tokens, eos_id=eos_id
            )

        # Token embeddings: [b, s] -> [b, s, d]
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # Pass through transformer layers
        for layer_id, layer in self.layers.items():
            # Pass freqs_cis for rotary embeddings
            h = layer(
                h,
                freqs_cis=self.freqs_cis,
            )

        # Final normalization: [b, s, d]
        h = self.norm(h) if self.norm else h

        # Output projection: [b, s, d] -> [b, s, vocab_size]
        output = self.output(h) if self.output else h

        return output


def build_modular_llama_model(model_args: TransformerModelArgs) -> ModularTransformer:
    """
    Builder function to create a modular Llama model from TransformerModelArgs.

    This function creates the individual components (embeddings, layers, norm, output)
    and then composes them into a ModularTransformer, following torchtune's pattern
    while maintaining torchtitan compatibility.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments

    Returns:
        ModularTransformer: The modular transformer model
    """
    # Create token embeddings
    tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

    # Create individual components for a transformer layer (torchtune style)

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
    layer_template = ModularTransformerBlock(
        attn=attention,
        mlp=feed_forward,
        sa_norm=attention_norm,
        mlp_norm=ffn_norm,
        layer_id=0,  # Template layer
        weight_init_std=weight_init_std,
    )

    # Set up attention function for torchtitan compatibility
    try:
        attention.sdpa = build_attention(
            model_args.use_flex_attn, model_args.attn_mask_type
        )
    except ValueError:
        pass

    # Create final normalization and output layers
    norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
    output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    # Create the modular transformer using torchtune's pattern
    model = ModularTransformer(
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


# Backward compatibility alias
def create_modular_llama_model(model_args: TransformerModelArgs) -> ModularTransformer:
    """
    Factory function to create a modular Llama model.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments

    Returns:
        ModularTransformer: The modular transformer model
    """
    return build_modular_llama_model(model_args)
