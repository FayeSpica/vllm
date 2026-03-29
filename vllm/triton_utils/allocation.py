# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import triton


def set_triton_allocator(device: torch.device):
    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty(size, device=device, dtype=torch.int8)

    # Standard API
    triton.set_allocator(alloc_fn)

    # Workaround for triton bug: jit.py imports _allocator by value
    # (from triton.runtime._allocation import _allocator), creating a
    # snapshot bound to None at import time. triton.set_allocator()
    # updates _allocation._allocator but NOT jit.py's local binding.
    # We must patch jit.py's module namespace directly.
    try:
        import triton.runtime.jit as _jit_module
        _jit_module._allocator = alloc_fn
    except (ImportError, AttributeError):
        pass
