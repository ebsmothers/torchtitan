# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader

from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from torchtitan.models.llama3.infra.pipeline import pipeline_llama
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .model.model import qwen3, Qwen3ModelArgs

__all__ = [
    "qwen3_configs",
]

qwen3_configs = {
    "32B": Qwen3ModelArgs(
        vocab_size=152064,
        dim=5120,
        intermediate_size=27392,
        n_layers=64,
        n_heads=40,
        n_kv_heads=40,
        max_seq_len=32768,
        rope_base=1000000,
        norm_eps=1e-6,
        use_scaled_rope=True,
    ),
}

register_train_spec(
    TrainSpec(
        name="qwen3",
        model_cls=qwen3,
        model_args=qwen3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=None,  # No custom adapter needed for now
    )
)
