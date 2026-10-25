"""Exercise 1: vector add.  out = x + y

Your job: fill in the body of `add_kernel`. The launch wrapper `add` is done for you.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each "program" (one thread block on the GPU) handles BLOCK_SIZE contiguous elements.
    #
    # TODO 1: which program am I?            -> tl.program_id(axis=0)
    # TODO 2: which elements do I own?       -> a block of offsets starting at pid * BLOCK_SIZE
    #                                           (tl.arange(0, BLOCK_SIZE) gives 0..BLOCK_SIZE-1)
    # TODO 3: the last block may run past n_elements. Build a boolean mask.
    # TODO 4: load x and y with the mask, add, store to out with the mask.
    pass


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.shape == y.shape and x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    # One program per BLOCK_SIZE elements, rounded up.
    grid = (triton.cdiv(n, block_size),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out
