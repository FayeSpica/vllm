# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""V100-compatible Triton MLA decode attention kernel.

The standard Triton MLA decode kernel crashes with illegal memory access
on SM 7.0 (V100).  This module provides a simplified kernel that:
- Tiles the D dimension in chunks of 64 (avoids padding to 1024)
- Uses conservative settings (BLOCK_N=16, num_warps=2, num_stages=1)
- Single-pass online softmax (no split-K, no stage2 merge)
- Direct paged KV cache access (no gather/copy step)
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _volta_mla_decode_kernel(
    Q,              # [B, H, D]
    KV_cache,       # [num_blocks, PAGE_SIZE, D]
    O,              # [B, H, D_NOPE]
    LSE,            # [B, H]
    block_table,    # [B, max_pages]
    seq_lens,       # [B]
    sm_scale,       # float
    # Strides (in elements, not bytes)
    stride_qb, stride_qh,
    stride_kvb, stride_kvp,
    stride_ob, stride_oh,
    stride_btb,
    # Constexpr dimensions
    H: tl.constexpr,
    D: tl.constexpr,           # full KV dim (e.g. 576)
    D_NOPE: tl.constexpr,      # value/output dim (e.g. 512)
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,     # sequence block size
    TILE_KD: tl.constexpr,     # K dimension tile size for score computation
):
    """Single-pass flash-decoding for MLA on V100.

    Grid: (batch, num_heads)
    Each program computes attention for one (batch, head) pair,
    iterating over the full KV sequence with online softmax.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    cur_seq_len = tl.load(seq_lens + pid_b)
    if cur_seq_len == 0:
        return

    # Base pointers
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    bt_base = block_table + pid_b * stride_btb
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    # V dimension offsets (D_NOPE is power of 2, e.g. 512)
    offs_v = tl.arange(0, D_NOPE)

    # Online softmax state
    m_i = tl.full([1], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D_NOPE], dtype=tl.float32)

    # Main loop: iterate over KV sequence in blocks of BLOCK_N
    for start_n in range(0, cur_seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cur_seq_len

        # --- Page table lookup ---
        page_idx = offs_n // PAGE_SIZE
        page_off = offs_n % PAGE_SIZE
        phys_page = tl.load(bt_base + page_idx, mask=mask_n, other=0)
        # Base offset for each KV position [BLOCK_N]
        kv_off = phys_page * stride_kvb + page_off * stride_kvp

        # --- Phase 1: Compute attention scores ---
        # Tile over D to avoid padding 576 -> 1024
        scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d_start in range(0, D, TILE_KD):
            d_offs = d_start + tl.arange(0, TILE_KD)
            d_mask = d_offs < D

            # Load Q tile [TILE_KD]
            q_tile = tl.load(
                q_base + d_offs, mask=d_mask, other=0.0
            ).to(tl.float32)

            # Load K tile [BLOCK_N, TILE_KD] from paged cache
            k_ptrs = KV_cache + kv_off[:, None] + d_offs[None, :]
            k_tile = tl.load(
                k_ptrs,
                mask=mask_n[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Partial dot product
            scores += tl.sum(q_tile[None, :] * k_tile, axis=1)

        scores = scores * sm_scale
        scores = tl.where(mask_n, scores, float('-inf'))

        # --- Phase 2: Online softmax ---
        m_block = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(mask_n, p, 0.0)
        l_new = alpha * l_i + tl.sum(p, axis=0)

        # Rescale running accumulator
        acc = acc * alpha

        # --- Phase 3: Weighted value accumulation ---
        # V = KV_cache[:, :D_NOPE], load [BLOCK_N, D_NOPE]
        v_ptrs = KV_cache + kv_off[:, None] + offs_v[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # p[BLOCK_N] @ V[BLOCK_N, D_NOPE] -> [D_NOPE]
        acc += tl.sum(p[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    # Normalize
    acc = acc / l_i

    # Store output [D_NOPE]
    tl.store(o_base + offs_v, acc.to(O.dtype.element_ty))

    # Store log-sum-exp (scalar per batch,head)
    lse_val = m_i + tl.log(l_i)
    tl.store(LSE + pid_b * H + pid_h, lse_val.to(LSE.dtype.element_ty))


def volta_mla_decode_fwd(
    q: torch.Tensor,              # [B, H, D]
    kv_cache: torch.Tensor,       # [num_blocks, PAGE_SIZE, D]
    o: torch.Tensor,              # [B, H, D_NOPE]
    lse: torch.Tensor,            # [B, H]
    block_table: torch.Tensor,    # [B, max_pages]
    seq_lens: torch.Tensor,       # [B]
    scale: float,
    page_size: int,
) -> None:
    """Launch V100-compatible MLA decode attention kernel."""
    B, H, D = q.shape
    D_NOPE = o.shape[-1]

    if B == 0:
        return

    BLOCK_N = 16
    TILE_KD = 64

    grid = (B, H)

    _volta_mla_decode_kernel[grid](
        q, kv_cache, o, lse, block_table, seq_lens,
        scale,
        q.stride(0), q.stride(1),
        kv_cache.stride(0), kv_cache.stride(1),
        o.stride(0), o.stride(1),
        block_table.stride(0),
        H=H, D=D, D_NOPE=D_NOPE,
        PAGE_SIZE=page_size,
        BLOCK_N=BLOCK_N,
        TILE_KD=TILE_KD,
        num_warps=2,
        num_stages=1,
    )
