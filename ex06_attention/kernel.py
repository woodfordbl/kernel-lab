"""Exercise 6, Part B: FlashAttention forward pass.

    O = softmax(Q K^T * scale) V          Q, K, V, O: [B, H, N, D]

Each program owns BLOCK_M query rows of one (batch, head). It keeps its Q tile resident,
streams K and V in BLOCK_N chunks, and never writes the N x N score matrix anywhere.
The trick that makes that possible is the online softmax, with three running values per
query row:
    m  the largest score seen so far
    l  the sum of exp(score - m) seen so far
    acc  the sum of exp(score - m) * v seen so far (unnormalized output)
When a new chunk raises the max from m to m_new, everything accumulated so far was scaled
by exp(-m) and must be rescaled by alpha = exp(m - m_new). At the end, O = acc / l.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, N, scale,
                          stride_bh, stride_n,           # Q, K, V, O share one layout
                          D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)   # which block of queries
    bh = tl.program_id(1)      # which (batch, head)
    base = bh * stride_bh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + base + offs_m[:, None] * stride_n + offs_d[None, :],
                mask=offs_m[:, None] < N, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # With causal masking, keys after this block's last query are never needed.
    hi = tl.minimum((pid_m + 1) * BLOCK_M, N) if CAUSAL else N

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # TODO 1: load the K tile as [D, BLOCK_N] (already transposed for the dot) and the
        #         V tile as [BLOCK_N, D]. Mask keys past N.
        # TODO 2: s = tl.dot(q, k) * scale   -> [BLOCK_M, BLOCK_N] scores, in registers only
        # TODO 3: invalid scores -> -inf: keys past N, and with CAUSAL, keys after the query
        #         (offs_n[None, :] > offs_m[:, None]). tl.where(cond, s, float("-inf"))
        # TODO 4: online softmax update:
        #           m_new = max(m_i, rowmax(s))
        #           p = exp(s - m_new[:, None])
        #           alpha = exp(m_i - m_new)
        #           l_i = l_i * alpha + rowsum(p)
        #           acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        #           m_i = m_new
        pass

    # TODO 5: normalize (acc / l_i) and store the O tile, masking rows past N.


def flash_attention(q, k, v, causal: bool = False, block_m: int = 64, block_n: int = 64):
    B, H, N, D = q.shape
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert D in (16, 32, 64, 128)
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, block_m), B * H)
    flash_attn_fwd_kernel[grid](q, k, v, o, N, D**-0.5, N * D, D,
                                D=D, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,
                                num_warps=4, num_stages=2)
    return o
