"""Exercise 7: the fused-MoE grouped GEMM.

One launch computes every expert's projection:
    out[pair] = x[token(pair)] @ W[expert(pair)]        (optionally * router weight)
for every (token, expert) pair, where pair = token * top_k + j.

The rows arrive pre-sorted by expert (see align.py), so each BLOCK_M-row tile belongs to one
expert. The kernel never copies x into sorted order: it GATHERS the rows it needs while
loading the A tile, using sorted_ids. That fuses the "permute" step into the GEMM.
"""
import torch
import torch.nn.functional as Fn
import triton
import triton.language as tl

from ex07_moe.align import moe_align


@triton.jit
def moe_gemm_kernel(x_ptr, w_ptr, out_ptr, sorted_ids_ptr, expert_ids_ptr, topk_w_ptr,
                    num_pairs, N, K, top_k,
                    stride_xm, stride_xk, stride_we, stride_wk, stride_wn, stride_om, stride_on,
                    MUL_ROUTED_WEIGHT: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n          # which block of sorted rows
    pid_n = pid % num_pid_n
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # TODO 1: load this block's BLOCK_M entries of sorted_ids -> pair ids.
    #         Padding entries equal num_pairs: build row_mask = pair < num_pairs.
    #         The token each pair came from is pair // top_k. That's the row of x to gather.
    # TODO 2: load this block's expert id (one scalar) from expert_ids_ptr.
    #         Cast it with .to(tl.int64) before multiplying by stride_we: the expert offset
    #         can exceed 2^31 elements on big models.
    # TODO 3: build A pointers (gathered rows of x) and B pointers (that expert's [K, N] slice
    #         of w), then the usual K loop with tl.dot. Mask A rows with row_mask.
    # TODO 4: epilogue. If MUL_ROUTED_WEIGHT, multiply each row by its router weight
    #         topk_w[pair]. Store row r of the tile to out[pair_r] (a scatter), masked.
    pass


def moe_gemm(x, w, topk_ids, topk_weights, sorted_ids, expert_ids, mul_routed_weight,
             block_m, block_n=64, block_k=32):
    """x: [rows, K]; w: [E, K, N]. Each pair p reads row p // top_k of x and writes out[p]."""
    T, top_k = topk_ids.shape
    E, K, N = w.shape
    out = torch.zeros((T * top_k, N), device=x.device, dtype=x.dtype)
    grid = (expert_ids.numel() * triton.cdiv(N, block_n),)
    moe_gemm_kernel[grid](x, w, out, sorted_ids, expert_ids, topk_weights,
                          T * top_k, N, K, top_k,
                          x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2),
                          out.stride(0), out.stride(1),
                          MUL_ROUTED_WEIGHT=mul_routed_weight,
                          BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4)
    return out


def moe_mlp(x, w13, w2, router_logits, top_k, block_m=64):
    """A full MoE MLP. x: [T, D]; w13: [E, D, 2F] (gate | up); w2: [E, F, D]."""
    T, D = x.shape
    E = w13.shape[0]
    probs = torch.softmax(router_logits.float(), dim=-1)
    topk_w, topk_ids = torch.topk(probs, top_k, dim=-1)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(x.dtype).contiguous()
    sorted_ids, expert_ids = moe_align(topk_ids, E, block_m)

    # GEMM 1 gathers token rows: pair p reads x[p // top_k].
    h = moe_gemm(x, w13, topk_ids, topk_w, sorted_ids, expert_ids, False, block_m)
    F = w13.shape[2] // 2
    h = Fn.silu(h[:, :F]) * h[:, F:]          # stretch: fuse this into GEMM 1 (exercise 5)

    # GEMM 2 reads row p of h directly (h is already one row per pair), so call it as if
    # top_k = 1: then token(p) = p. The sentinel T * top_k and the sorted order are unchanged.
    pairs_as_tokens = topk_ids.reshape(T * top_k, 1)
    y = moe_gemm(h.contiguous(), w2, pairs_as_tokens, topk_w.view(-1, 1), sorted_ids,
                 expert_ids, True, block_m)
    return y.view(T, top_k, D).sum(dim=1)     # combine: rows are already weighted, so just sum
