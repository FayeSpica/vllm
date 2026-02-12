# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.gptq_triton import (
    GPTQ_TRITON_SUPPORTED_GROUP_SIZES,
    gptq_gemm_triton,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    permute_param_layout_,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import HAS_TRITON


def _reorder_packed_qweight_by_perm(
    qweight: torch.Tensor, perm: torch.Tensor, bits: int
) -> torch.Tensor:
    """Reorder packed quantized weight rows along the K dimension.

    This performs the same row reordering as make_sequential_Xbit_kernel in
    the CUDA code, but without the subsequent bit-level shuffle that
    shuffle_Xbit_kernel would apply.

    Args:
        qweight: Packed weight tensor [K // pack_factor, N], int32.
        perm: Permutation array (argsort of g_idx), length K.
        bits: Number of bits per weight element (e.g. 4).
    Returns:
        New qweight tensor with rows reordered according to perm.
    """
    assert bits == 4, "Only 4-bit packing is supported"
    pack_factor = 32 // bits  # 8 for 4-bit
    K_packed, N = qweight.shape
    K = K_packed * pack_factor

    assert perm.shape[0] == K

    # Unpack: extract each 4-bit value into its own row -> [K, N] uint8
    # qweight[k // 8, n] >> ((k % 8) * 4) & 0xF
    k_indices = torch.arange(K, device=qweight.device)
    packed_row = k_indices // pack_factor  # [K]
    bit_offset = (k_indices % pack_factor) * bits  # [K]

    # Gather packed rows: [K, N]
    packed_vals = qweight[packed_row]  # [K, N]
    # Extract individual values
    unpacked = (packed_vals >> bit_offset[:, None]) & ((1 << bits) - 1)

    # Reorder along K dimension
    unpacked = unpacked[perm.long()]

    # Repack: pack 8 consecutive values into each int32
    unpacked = unpacked.to(torch.int32)
    new_qweight = torch.zeros(
        (K_packed, N), dtype=torch.int32, device=qweight.device
    )
    for i in range(pack_factor):
        new_qweight |= unpacked[i::pack_factor] << (i * bits)

    return new_qweight


class TritonLinearKernel(MPLinearKernel):
    """Triton kernel for GPTQ W4A16 on compute capability >= 7.0
    (Volta and above)."""

    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(
        cls, c: MPLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return (
                False,
                "GPTQ Triton is only supported on CUDA and ROCm",
            )
        capability_tuple = current_platform.get_device_capability()
        if capability_tuple is None or capability_tuple.to_int() < 70:
            return False, "GPTQ Triton requires compute capability >= 7.0"
        if not HAS_TRITON:
            return False, "Triton is not available"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                "GPTQ Triton only supports GPTQ W4A16 "
                f"(weights {cls.SUPPORTED_QUANT_TYPES})",
            )
        if c.zero_points:
            return (
                False,
                "GPTQ Triton only supports symmetric GPTQ W4A16",
            )
        if c.act_type != torch.float16:
            return (
                False,
                "GPTQ Triton only supports float16 activations (W4A16)",
            )
        if c.out_type is not None and c.out_type != torch.float16:
            return (
                False,
                "GPTQ Triton only supports float16 outputs (W4A16)",
            )
        if c.has_g_idx and (
            c.partition_weight_shape[0] != c.full_weight_shape[0]
        ):
            return (
                False,
                "Act reordering not supported when input features "
                "are partitioned across devices",
            )
        if c.partition_weight_shape[0] % 8 != 0:
            return (
                False,
                "Input features must be divisible by 8 for "
                "GPTQ 4-bit packing",
            )
        if (
            c.group_size != -1
            and c.full_weight_shape[0] % c.group_size != 0
        ):
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide "
                f"the number of input features "
                f"({c.full_weight_shape[0]})",
            )
        if c.group_size not in GPTQ_TRITON_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Unsupported group_size {c.group_size} for GPTQ Triton",
            )
        return True, None

    def process_weights_after_loading(
        self, layer: torch.nn.Module
    ) -> None:
        c = self.config

        if c.has_g_idx:

            def transform_w_g_idx(x):
                # Convert group indices to permutation array
                return torch.argsort(x).to(torch.int)

            self._transform_param(
                layer, self.w_gidx_name, transform_w_g_idx
            )
        else:
            self.w_gidx_name = "weight_g_idx"
            device = getattr(layer, self.w_q_name).device
            empty_g_idx = torch.nn.Parameter(
                torch.empty((0,), dtype=torch.int, device=device),
                requires_grad=False,
            )
            setattr(layer, self.w_gidx_name, empty_g_idx)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            assert self.w_gidx_name is not None
            g_idx = getattr(layer, self.w_gidx_name)

            permute_param_layout_(
                x, input_dim=0, output_dim=1, packed_dim=0
            )
            x_cont = x.data.contiguous()
            if g_idx.numel() > 0:
                # Reorder weight rows along K dimension according to
                # g_idx permutation.  We do NOT call ops.gptq_shuffle()
                # because it also applies a bit-level shuffle
                # (shuffle_4bit_8) designed for the Exllama CUDA kernel's
                # deq2 LUT. The Triton kernel expects the original GPTQ
                # packing where k-th 4-bit value is at bits [k*4+3 : k*4]
                # within each int32, so only the row reordering is needed.
                x_cont = _reorder_packed_qweight_by_perm(
                    x_cont, g_idx, c.weight_type.size_bits
                )
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, _, w_g_idx = self._get_weight_params(layer)

        if w_g_idx is not None and w_g_idx.numel() > 0:
            x_2d = x_2d[:, w_g_idx.to(torch.long)]

        output = gptq_gemm_triton(
            x_2d, w_q, w_s, c.group_size, split_k_iters=1
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
