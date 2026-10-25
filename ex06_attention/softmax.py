"""Exercise 6, Part A: row-wise softmax in one kernel.

softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

Subtracting the row max doesn't change the answer, but it keeps exp() from overflowing.
This is the warm-up for FlashAttention, which does the same thing in pieces.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, N, stride_row, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    # TODO 1: load the row in fp32. Padding lanes must not win the max or add to the sum:
    #         what `other=` value makes exp(other - max) equal 0?
    # TODO 2: m = row max, p = exp(x - m), l = sum(p)
    # TODO 3: store p / l
    pass


def softmax(x: torch.Tensor) -> torch.Tensor:
    R, N = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(R,)](x, out, N, x.stride(0), BLOCK_N=triton.next_power_of_2(N))
    return out
