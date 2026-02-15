# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""V100-compatible Triton MLA decode attention kernel with split-K.

The standard Triton MLA decode kernel crashes with illegal memory access
on SM 7.0 (V100).  This module provides a simplified kernel that:
- Tiles the D dimension in chunks of 64 (avoids padding to 1024)
- Uses conservative settings (BLOCK_N=16, num_warps=2, num_stages=1)
- Split-K to maximize SM utilization (V100 has 80 SMs)
- Direct paged KV cache access (no gather/copy step)
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _volta_mla_decode_stage1(
    Q,              # [B, H, D]
    KV_cache,       # [num_blocks, PAGE_SIZE, D]
    Attn_logits,    # [B, H, NUM_KV_SPLITS, D_NOPE + 1]  fp32
    block_table,    # [B, max_pages]
    seq_lens,       # [B]
    sm_scale,       # float
    # Strides
    stride_qb, stride_qh,
    stride_kvb, stride_kvp,
    stride_alb, stride_alh, stride_als,
    stride_btb,
    # Constexpr dimensions
    H: tl.constexpr,
    D: tl.constexpr,
    D_NOPE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_KD: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    """Stage 1: Each program handles one (batch, head, split) partition.

    Grid: (batch, num_heads, NUM_KV_SPLITS)
    Outputs normalized partial attention + LSE to Attn_logits buffer.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    split_id = tl.program_id(2)

    cur_seq_len = tl.load(seq_lens + pid_b)

    # Compute this split's range
    kv_per_split = (cur_seq_len + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS
    split_start = split_id * kv_per_split
    split_end = tl.minimum(split_start + kv_per_split, cur_seq_len)

    # Output base pointer in attn_logits
    al_base = (Attn_logits + pid_b * stride_alb
               + pid_h * stride_alh + split_id * stride_als)
    offs_v = tl.arange(0, D_NOPE)

    # If this split has no work, store -inf LSE and zero acc
    if split_start >= cur_seq_len:
        tl.store(al_base + offs_v, tl.zeros([D_NOPE], dtype=tl.float32))
        tl.store(al_base + D_NOPE + tl.arange(0, 1),
                 tl.full([1], float('-inf'), dtype=tl.float32))
        return

    # Base pointers
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    bt_base = block_table + pid_b * stride_btb

    # Online softmax state
    m_i = tl.full([1], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D_NOPE], dtype=tl.float32)

    for start_n in range(split_start, split_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < split_end

        # Page table lookup
        page_idx = offs_n // PAGE_SIZE
        page_off = offs_n % PAGE_SIZE
        phys_page = tl.load(bt_base + page_idx, mask=mask_n, other=0)
        kv_off = phys_page * stride_kvb + page_off * stride_kvp

        # Score computation with D-tiling
        scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d_start in range(0, D, TILE_KD):
            d_offs = d_start + tl.arange(0, TILE_KD)
            d_mask = d_offs < D

            q_tile = tl.load(
                q_base + d_offs, mask=d_mask, other=0.0
            ).to(tl.float32)

            k_ptrs = KV_cache + kv_off[:, None] + d_offs[None, :]
            k_tile = tl.load(
                k_ptrs,
                mask=mask_n[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            scores += tl.sum(q_tile[None, :] * k_tile, axis=1)

        scores = scores * sm_scale
        scores = tl.where(mask_n, scores, float('-inf'))

        # Online softmax
        m_block = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(mask_n, p, 0.0)
        l_new = alpha * l_i + tl.sum(p, axis=0)

        acc = acc * alpha

        # V accumulation
        v_ptrs = KV_cache + kv_off[:, None] + offs_v[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    # Normalize partial output
    acc = acc / l_i

    # Store normalized partial output [D_NOPE] + LSE [1]
    tl.store(al_base + offs_v, acc)
    lse_val = m_i + tl.log(l_i)
    tl.store(al_base + D_NOPE + tl.arange(0, 1), lse_val)


@triton.jit
def _volta_mla_decode_stage2(
    Attn_logits,    # [B, H, NUM_KV_SPLITS, D_NOPE + 1]  fp32
    O,              # [B, H, D_NOPE]
    LSE,            # [B, H]
    stride_alb, stride_alh, stride_als,
    stride_ob, stride_oh,
    H: tl.constexpr,
    D_NOPE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    """Stage 2: Merge partial attention results from all splits.

    Grid: (batch, num_heads)
    Uses online softmax merge with LSE for numerical stability.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    al_base = Attn_logits + pid_b * stride_alb + pid_h * stride_alh
    offs_v = tl.arange(0, D_NOPE)

    # Online merge state
    m_i = tl.full([1], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D_NOPE], dtype=tl.float32)

    for split_id in range(NUM_KV_SPLITS):
        split_base = al_base + split_id * stride_als

        # Load LSE for this split
        lse_k = tl.load(split_base + D_NOPE + tl.arange(0, 1))
        # Load normalized partial output
        o_k = tl.load(split_base + offs_v)

        # Online merge: weight each split by exp(lse_k)
        m_new = tl.maximum(m_i, lse_k)
        old_scale = tl.exp(m_i - m_new)
        new_scale = tl.exp(lse_k - m_new)

        acc = acc * old_scale + o_k * new_scale
        l_i = l_i * old_scale + new_scale
        m_i = m_new

    # Normalize
    acc = acc / l_i

    # Store final output
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    tl.store(o_base + offs_v, acc.to(O.dtype.element_ty))

    # Store final LSE
    lse_val = (m_i + tl.log(l_i)).to(LSE.dtype.element_ty)
    lse_ptrs = LSE + pid_b * H + pid_h + tl.arange(0, 1)
    tl.store(lse_ptrs, lse_val)


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

    # Split-K: target ~80 blocks for V100's 80 SMs
    total_blocks = B * H
    NUM_KV_SPLITS = min(4, max(1, 80 // max(total_blocks, 1)))

    # Allocate intermediate buffer for split-K merge
    attn_logits = torch.empty(
        (B, H, NUM_KV_SPLITS, D_NOPE + 1),
        dtype=torch.float32,
        device=q.device,
    )

    # Stage 1: compute partial attention per split
    _volta_mla_decode_stage1[(B, H, NUM_KV_SPLITS)](
        q, kv_cache, attn_logits, block_table, seq_lens,
        scale,
        q.stride(0), q.stride(1),
        kv_cache.stride(0), kv_cache.stride(1),
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        block_table.stride(0),
        H=H, D=D, D_NOPE=D_NOPE,
        PAGE_SIZE=page_size,
        BLOCK_N=BLOCK_N,
        TILE_KD=TILE_KD,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        num_warps=2,
        num_stages=1,
    )

    # Stage 2: merge partial results
    _volta_mla_decode_stage2[(B, H)](
        attn_logits, o, lse,
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        o.stride(0), o.stride(1),
        H=H, D_NOPE=D_NOPE,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        num_warps=2,
        num_stages=1,
    )
