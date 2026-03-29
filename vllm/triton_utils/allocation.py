# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import triton


def set_triton_allocator(device: torch.device):
    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty(size, device=device, dtype=torch.int8)

    # Standard API
    triton.set_allocator(alloc_fn)

    # Workaround for triton >= 3.6 where _allocator is a ContextVar.
    # ContextVar values are per-context, so set_allocator() only sets
    # the value in the current context. torch.compile / CUDA graph
    # execution may run in a different context where the ContextVar
    # has its default (NullAllocator that raises RuntimeError).
    # Fix: monkey-patch NullAllocator.__call__ so the default works.
    try:
        from triton.runtime._allocation import NullAllocator
        NullAllocator.__call__ = lambda self, size, alignment, stream: \
            alloc_fn(size, alignment, stream)
    except ImportError:
        pass

    # Workaround for triton < 3.6 where _allocator is a plain variable
    # imported by value in jit.py (snapshot bound to None at import time).
    try:
        import triton.runtime.jit as _jit_module
        if hasattr(_jit_module, '_allocator'):
            _jit_module._allocator = alloc_fn
    except ImportError:
        pass
