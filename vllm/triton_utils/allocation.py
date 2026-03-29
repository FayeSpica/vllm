# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import triton


def set_triton_allocator(device: torch.device):
    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty(size, device=device, dtype=torch.int8)

    # Try the standard API first
    triton.set_allocator(alloc_fn)

    # Also directly patch the runtime allocator as a fallback,
    # in case triton.set_allocator doesn't propagate correctly
    # (e.g. in distributed environments where CUDA_VISIBLE_DEVICES
    # is temporarily empty during triton import).
    try:
        from triton.runtime import _allocation
        _allocation._allocator = alloc_fn
    except (ImportError, AttributeError):
        pass
