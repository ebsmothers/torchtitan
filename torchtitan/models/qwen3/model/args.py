from dataclasses import dataclass
from typing import Optional

from torch import nn

from torchtitan.config import JobConfig

from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class Qwen3ModelArgs(BaseModelArgs):
    """Model arguments for Qwen3 32B model."""

    vocab_size: int = 152064
    dim: int = 5120
    intermediate_size: int = 27392
    n_layers: int = 64
    n_heads: int = 40
    n_kv_heads: int = 40
    max_seq_len: int = 32768
    rope_base: int = 1000000
    rope_theta: float = 1000000.0  # Same as rope_base for compatibility
    norm_eps: float = 1e-6
    use_scaled_rope: bool = True
    # Additional attributes commonly expected by training infrastructure
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_scaling: Optional[dict] = None

    # TODO: check these
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0
    depth_init: bool = True

    # Copy-paste from Llama3 for now
    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

        if (
            job_config.parallelism.pipeline_parallel_degree > 1
            and self.use_flex_attn
            and self.attn_mask_type == "block_causal"
        ):
            raise RuntimeError(
                "PP + block causal FlexAttention support will be fixed soon."
            )
        self.max_seq_len = seq_len

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        print("HIHIHI", nparams, num_flops_per_token)
        return nparams, num_flops_per_token
