"""Exercise 3: fused residual add + RMSNorm.

    h   = x + residual                      (the new residual stream)
    out = h / sqrt(mean(h^2) + eps) * weight

PyTorch eager does this in about 5 kernels. You'll do it in one: each program owns one row
(one token's hidden vector), keeps it in registers, reduces it, and writes two outputs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, h_ptr,
                       D, stride_row, eps,
                       BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    # TODO 1: load row `row` of x and residual (fp32), add them -> h. Use stride_row.
    #         Store h to h_ptr (cast back to the output dtype). This is the new residual.
    # TODO 2: mean of squares over the row. tl.sum(v, axis=0) reduces a vector to a scalar.
    #         Masked lanes must contribute 0. What did tl.load put there? (see `other=`)
    #         Divide by D, not BLOCK_D.
    # TODO 3: r = 1 / sqrt(ms + eps)  (tl.rsqrt exists). Load weight, compute h * r * w, store.
    pass


def add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """x, residual: [T, D] contiguous. weight: [D]. Returns (out, h)."""
    T, D = x.shape
    assert x.is_contiguous() and residual.is_contiguous()
    out = torch.empty_like(x)
    h = torch.empty_like(x)
    BLOCK_D = triton.next_power_of_2(D)  # tl.arange needs a power of 2
    num_warps = 4 if BLOCK_D <= 2048 else 8
    add_rmsnorm_kernel[(T,)](x, residual, weight, out, h, D, x.stride(0), eps,
                             BLOCK_D=BLOCK_D, num_warps=num_warps)
    return out, h
