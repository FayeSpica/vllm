# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

logger = init_logger(__name__)

# V100 (SM 7.0): Triton MLA decode kernel OOMs on launch because V100
# VRAM is fully occupied by model weights + KV cache, leaving no room
# for Triton JIT compilation buffers.  Use PyTorch fallback instead.
_USE_VOLTA_DECODE = (
    current_platform.is_cuda()
    and current_platform.get_device_capability() == (7, 0)
)


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonMLA V1 with FP8 KV cache not yet supported"
            )

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        if get_flash_attn_version() is None:
            raise RuntimeError(
                "FlashAttention is not available on this device. "
                "MLA prefill should use SDPA path."
            )
        return super()._flash_attn_varlen_diff_headdims(
            q,
            k,
            v,
            return_softmax_lse=return_softmax_lse,
            softmax_scale=softmax_scale,
            **kwargs,
        )

    def _volta_forward_mqa(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Vectorized PyTorch MLA decode attention for V100 (SM 7.0).

        The Triton MLA decode kernel OOMs on V100 because VRAM is fully
        occupied by model weights + KV cache.  This fallback uses batched
        PyTorch matmul (leveraging cuBLAS / tensor cores) with fp32 softmax.
        """
        B, H, D = q.shape
        D_nope = self.kv_lora_rank

        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        max_sl = seq_lens.max().item()
        if max_sl == 0:
            o = torch.zeros(B, H, D_nope, dtype=q.dtype, device=q.device)
            lse = torch.zeros(B, H, dtype=q.dtype, device=q.device)
            return o, lse

        # Gather all KV cache pages in one indexing op
        max_num_pages = (max_sl + PAGE_SIZE - 1) // PAGE_SIZE
        page_indices = block_table[:, :max_num_pages]  # [B, max_pages]
        kv = kv_c_and_k_pe_cache[page_indices]  # [B, max_pages, PAGE_SIZE, D]
        kv = kv.reshape(B, -1, D)[:, :max_sl, :]  # [B, max_sl, D]

        # Attention mask for variable sequence lengths
        positions = torch.arange(
            max_sl, device=q.device).unsqueeze(0)  # [1, max_sl]
        # True where position should be masked (invalid)
        inv_mask = positions >= seq_lens.unsqueeze(1)  # [B, max_sl]

        # Scores: [B, H, D] @ [B, D, max_sl] → [B, H, max_sl]
        # Use fp16 matmul to leverage V100 tensor cores (125 vs 15.7 TFLOPS)
        scores = torch.matmul(q, kv.transpose(1, 2)).float() * self.scale
        scores.masked_fill_(inv_mask.unsqueeze(1), float('-inf'))

        # Numerically stable softmax in float32
        max_s = scores.max(dim=-1, keepdim=True).values  # [B, H, 1]
        exp_s = torch.exp(scores - max_s)
        exp_s.masked_fill_(inv_mask.unsqueeze(1), 0.0)
        sum_exp = exp_s.sum(dim=-1, keepdim=True)  # [B, H, 1]
        attn_w = exp_s / sum_exp  # [B, H, max_sl]

        # Output: [B, H, max_sl] @ [B, max_sl, D_nope] → [B, H, D_nope]
        # Cast attn_w back to fp16 for tensor core matmul
        v = kv[:, :, :D_nope]
        o = torch.matmul(attn_w.to(q.dtype), v)

        # LSE for merge with prefill
        lse = (max_s.squeeze(-1) + torch.log(
            sum_exp.squeeze(-1))).to(q.dtype)  # [B, H]

        return o, lse

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 Triton MLA not yet supported")

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)

        if _USE_VOLTA_DECODE:
            return self._volta_forward_mqa(
                q, kv_c_and_k_pe_cache, attn_metadata)

        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        # For batch invariance, use only 1 split to ensure deterministic reduction
        num_kv_splits = 1 if vllm_is_batch_invariant() else 4

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                q_num_heads,
                num_kv_splits,
                # NOTE: the +1 stores the LogSumExp (LSE) that the stage2
                # kernel uses to merge partial attention outputs across splits.
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
        )

        return o, lse
