# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

_FROM_HF = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}


def get_mapped_key(key: str, mapping_dict: dict[str, str]) -> str:
    try:
        # Checks if there is a layer # in the key
        if any(k.isdigit() for k in key.split(".")):
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


# TODO: hardcoding for laziness
num_heads = 32
num_kv_heads = 8
dim = 4096
head_dim = dim // num_heads


class Llama3StateDictAdapter(StateDictAdapter):
    @staticmethod
    def to_hf(state_dict: dict[str, Any]) -> dict[str, Any]:
        converted_state_dict = {}
        inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}

        def _permute(t, n_heads):
            return (
                t.view(n_heads, head_dim // 2, 2, dim)
                .transpose(1, 2)
                .reshape((head_dim * n_heads), dim)
            )

        for key, value in state_dict.items():
            new_key = get_mapped_key(key, inverted_mapping_dict)
            if "q_proj" in key:
                value = _permute(value, num_heads)
            elif "k_proj" in key:
                value = _permute(value, num_kv_heads)
            converted_state_dict[new_key] = value

        return converted_state_dict

    @staticmethod
    def from_hf(hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        converted_state_dict = {}

        def _permute(t, n_heads):
            return (
                t.view(n_heads, 2, head_dim // 2, dim)
                .transpose(1, 2)
                .reshape((head_dim * n_heads), dim)
            )

        for key, value in hf_state_dict.items():
            if "rotary_emb.inv_freq" not in key:  # Skip loading the position embeddings
                new_key = get_mapped_key(key, _FROM_HF)
                if "q_proj" in key:
                    value = _permute(value, num_heads)
                elif "k_proj" in key:
                    value = _permute(value, num_kv_heads)

                converted_state_dict[new_key] = value
        return converted_state_dict
