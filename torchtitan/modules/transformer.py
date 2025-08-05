# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Callable, List, Optional, Union

import torch
from torch import nn

from torchtitan.modules.attention import _MaskType, MultiHeadAttention


class TransformerSelfAttentionLayer(nn.Module):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    This is a modular version adapted for torchtitan that maintains compatibility
    with existing parallelization functions while providing the flexibility
    of torchtune's modular design.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        # Use torchtitan naming conventions as primary for parameter compatibility
        self.attention = attn
        self.feed_forward = mlp
        self.attention_norm = sa_norm or nn.Identity()
        self.ffn_norm = mlp_norm or nn.Identity()

        # Keep torchtune-style aliases for API compatibility
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.sa_scale = sa_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token.
            freqs_cis (Optional[torch.Tensor]): Precomputed rotary embeddings.
            **kwargs (dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h = self.sa_norm(x)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos, freqs_cis=freqs_cis)
        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


class TransformerCrossAttentionLayer(nn.Module):
    """
    Cross attention Transformer layer following the same conventions as the TransformerSelfAttentionLayer.
    Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        ca_norm (Optional[nn.Module]): Normalization to be applied before cross-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        ca_scale (Optional[nn.Module]): Module to scale cross-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        ca_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        ca_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.ca_norm = ca_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.ca_scale = ca_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def _skip_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Some tokens in x may not attend to any encoder inputs
        due to the cross attention mask (encoder_mask). This results in
        a full row of the attention matrix being masked out.
        """
        # no skip_mask if no masking
        if mask is None:
            return None
        # negate mask and convert to boolean mask
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        # True where all elements in a row are True
        mask = torch.all(mask, dim=-1, keepdim=True)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder.
            encoder_mask (Optional[torch.Tensor]): Boolean tensor defining a relational matrix between
                tokens and encoder embeddings.
            **kwargs (dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Skip cross attention when no secondary input
        if encoder_input is None:
            return x

        # A mask of tokens (x) with no encoder_input
        skip_mask = self._skip_mask(encoder_mask)
        if encoder_mask is not None:
            # This unmasks the skipped rows to avoid NaNs in SDPA Softmax backward
            encoder_mask = encoder_mask.masked_fill(skip_mask, True)

        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        attn_out = self.attn(self.ca_norm(x), encoder_input, mask=encoder_mask)
        if skip_mask is not None:
            attn_out = attn_out.masked_fill(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.ca_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))
        if skip_mask is not None:
            mlp_out = mlp_out.masked_fill(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Return a list of ``n`` identical layers.

    Args:
        module (nn.Module): module to be cloned
        n (int): number of clones

    Returns:
        nn.ModuleList: list of ``n`` identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Llama2 architecture.

    This is a modular version adapted for torchtitan that maintains compatibility
    with existing parallelization functions while providing the flexibility
    of torchtune's modular design.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layers (Union[nn.Module, list[nn.Module], nn.ModuleList, nn.ModuleDict]): A single transformer Decoder layer, an
            nn.ModuleList/nn.ModuleDict of layers or a list of layers.
        max_seq_len (int): maximum sequence length the model will be run with
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value.
        head_dim (int): embedding dimension for each head in self-attention.
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (Union[nn.Linear, Callable]): Callable that applies a linear transformation to the output of
            the decoder.
        num_layers (Optional[int]): Number of Transformer Decoder layers, only define when
            layers is not a list.
        output_hidden_states (Optional[list[int]]): list of layers (indices) to include in the output

    Raises:
        AssertionError:
            If ``num_layers`` is set and layer is a list, **or**
            ``num_layers`` is not set and layer is an ``nn.Module``.
    """

    def __init__(
        self,
        *,
        tok_embeddings: nn.Embedding,
        layers: Union[nn.Module, List[nn.Module], nn.ModuleList, nn.ModuleDict],
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: Union[nn.Linear, Callable],
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        # Handle different layer input types
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
            layer_list = _get_clones(layers, num_layers)
            layers = nn.ModuleDict(
                {str(i): layer for i, layer in enumerate(layer_list)}
            )

        self.tok_embeddings = tok_embeddings
        self.layers: nn.ModuleDict = layers  # Always ModuleDict after conversion above
        self.norm = norm
        self.output = output
        self.output_hidden_states = output_hidden_states or []
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        # For torchtitan compatibility
        self.vocab_size = getattr(output, "out_features", None)
        self.n_layers = len(layers) if hasattr(layers, "__len__") else num_layers

    def forward(
        self,
        tokens: Optional[torch.Tensor],
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        eos_id: Optional[int] = None,
        input_batch: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (Optional[torch.Tensor]): input tensor with shape ``[b x s]``
            mask (Optional[_MaskType]): attention mask
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder.
            encoder_mask (Optional[torch.Tensor]): Boolean tensor defining a relational matrix between
                tokens and encoder embeddings.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token.
            input_embeds (Optional[torch.Tensor]): Pass these instead of tokens to short-circuit token embeddings
            freqs_cis (Optional[torch.Tensor]): Precomputed rotary embeddings.
            eos_id (Optional[int]): End of sequence token id (for torchtitan compatibility).
            input_batch (Optional[torch.Tensor]): The input batch (for torchtitan compatibility).

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: output tensor with shape ``[b x s x v]`` or a list of layer output tensors.
        """
        # Handle flex attention initialization if needed (torchtitan compatibility)
        if hasattr(self, "model_args") and getattr(
            self.model_args, "use_flex_attn", False
        ):
            try:
                from torchtitan.models.attention import init_attention_mask

                init_attention_mask(
                    input_batch if input_batch is not None else tokens, eos_id=eos_id
                )
            except ImportError:
                pass

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens) if input_embeds is None else input_embeds

        hidden = []
        for i, (layer_id, layer) in enumerate(self.layers.items()):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
                freqs_cis=freqs_cis,
            )

        if len(self.layers) in self.output_hidden_states:
            hidden.append(h)

        # shape: [b, s, d]
        h = self.norm(h) if self.norm else h

        # shape: [b, seq_len, out_dim]
        output = self.output(h) if self.output else h

        # Output list if hidden states are requested, otherwise just the output
        output = output if not hidden else [*hidden, output]
        return output
