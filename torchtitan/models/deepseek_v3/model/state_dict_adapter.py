# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# from typing import Any, Dict

# import regex as re
# import torch

# from torchtitan.protocols.state_dict_adapter import StateDictAdapter

# # TODO: THIS IS NOT CORRECT UNTIL WE ADD PERMUTE FOR ATTN PROJS

# _FROM_HF = {
#     "model.embed_tokens.weight": "tok_embeddings.weight",
#     "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
#     "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
#     "model.norm.weight": "norm.scale",
#     # attenion weights
#     "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
#     "model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attention.wq_a.weight",
#     "model.layers.{}.self_attn.q_a_layernorm.weight": "layers.{}.attention.q_norm.weight",
#     "model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attention.wq_b.weight",
#     "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "layers.{}.attention.wkv_a.weight",
#     "model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.attention.kv_norm.weight",
#     "model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.wkv_b.weight",
#     "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
#     # mlp non-expert weights
#     "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
#     "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
#     "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
#     # mlp MoE expert weights
#     "model.layers.{}.mlp.experts.gate_proj.weight": "layers.{}.moe.experts.w1",
#     "model.layers.{}.mlp.experts.up_proj.weight": "layers.{}.moe.experts.w3",
#     "model.layers.{}.mlp.experts.down_proj.weight": "layers.{}.moe.experts.w2",
#     # mlp MoE shared expert weights
#     "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_expert.w1",
#     "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_expert.w3",
#     "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_expert.w2",
#     # mlp MoE token router weights
#     "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
#     "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.moe.expert_bias",
#     "norm.weight": "norm.weight",
#     "lm_head.weight": "output.weight",
#     "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
# }


# def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
#     try:
#         # Checks if there is a layer # in the key
#         if any(k.isdigit() for k in key.split(".")):
#             # Replace all numbers with "{}" to create key for lookup
#             abstract_key = re.sub(r"(\.\d+)", ".{}", key)
#             # Find all numbers in the key in order
#             layer_nums = re.findall(r"\d+", key)
#             # if "moe.experts" in key:
#             #     torch.distributed.breakpoint()
#             new_key = mapping_dict[abstract_key]
#             # Format with all numbers
#             new_key = new_key.format(*layer_nums)
#         else:
#             new_key = mapping_dict[key]
#     except KeyError as e:
#         raise Exception(
#             f'Error converting the state dict. Found unexpected key: "{key}". '
#             "Please make sure you're loading a checkpoint with the right format. "
#         ) from e

#     return new_key


# N_EXPERTS = 64
# N_SHARED_EXPERTS = 2


# class DeepSeekV3StateDictAdapter(StateDictAdapter):
#     @staticmethod
#     def to_hf(state_dict: dict[str, Any]) -> dict[str, Any]:
#         pass
#         # converted_state_dict = {}
#         # loaded_experts = set()
#         # loaded_shared_experts = set()
#         # inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}
#         # for key, value in state_dict.items():
#         #     # if "moe.experts" in key:
#         #     #     torch.distributed.breakpoint()
#         #     new_key = get_mapped_key(key, inverted_mapping_dict)
#         #     layer_num = key.split(".")[1]
#         #     if "moe.experts" in key and layer_num not in loaded:
#         #         loaded_experts.add(layer_num)
#         #         if len(loaded_experts) == N_EXPERTS:
#         #             load.update()
#         #     elif "moe.shared_expert" in key:
#         #         keys = [new_key.replace(".experts.", f".experts.{i}.") for i in range(N_SHARED_EXPERTS)]
#         #         value = torch.stack([])
#         #         converted_state_dict[new_key] = value
#         #         load.update()
#         #     else:
#         #         converted_state_dict[new_key] = value

#         # return converted_state_dict

#     @staticmethod
#     def from_hf(hf_state_dict: dict[str, Any]) -> dict[str, Any]:
#         converted_state_dict = {}
#         for key, value in hf_state_dict.items():
#             # Skip keys that should be ignored (like rotary embeddings)
#             if "rotary_emb.inv_freq" in key:
#                 continue

#             new_key = get_mapped_key(key, _FROM_HF)
#             converted_state_dict[new_key] = value
#         return converted_state_dict
