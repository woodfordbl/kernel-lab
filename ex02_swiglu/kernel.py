"""Exercise 2: fused SwiGLU activation.  out = SiLU(gate) * up,  SiLU(g) = g * sigmoid(g)

Part A: gate and up are separate contiguous [M, F] tensors. Treat them as flat 1-D arrays.
Part B: gate and up are packed side by side in one [M, 2F] tensor (gate = gu[:, :F],
        up = gu[:, F:]), the layout a fused gate+up GEMM produces. This needs 2-D indexing.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- Part A
@triton.jit
def swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # TODO: same skeleton as exercise 1 (pid, offsets, mask, load).
    # TODO: compute in fp32 for accuracy: g = tl.load(...).to(tl.float32)
    #       tl.sigmoid exists. Cast the result back before storing:
    #       .to(out_ptr.dtype.element_ty)
    pass


def swiglu(gate: torch.Tensor, up: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    swiglu_kernel[(triton.cdiv(n, block_size),)](gate, up, out, n, BLOCK_SIZE=block_size)
    return out


# ---------------------------------------------------------------- Part B
@triton.jit
def swiglu_packed_kernel(gu_ptr, out_ptr, M, F, stride_gu_row, stride_out_row,
                         BLOCK_M: tl.constexpr, BLOCK_F: tl.constexpr):
    # A 2-D grid: program (pid_m, pid_f) owns a BLOCK_M x BLOCK_F tile of `out`.
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)
    # TODO: build row offsets (length BLOCK_M) and column offsets (length BLOCK_F).
    # TODO: turn them into a 2-D tile of pointers with broadcasting:
    #       rows[:, None] * stride + cols[None, :]
    # TODO: the up value for column c lives F columns to the right of its gate value.
    # TODO: the mask is 2-D too: (rows < M)[:, None] & (cols < F)[None, :]
    pass


def swiglu_packed(gu: torch.Tensor, block_m: int = 32, block_f: int = 128) -> torch.Tensor:
    M, two_f = gu.shape
    F = two_f // 2
    out = torch.empty((M, F), device=gu.device, dtype=gu.dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(F, block_f))
    swiglu_packed_kernel[grid](gu, out, M, F, gu.stride(0), out.stride(0),
                               BLOCK_M=block_m, BLOCK_F=block_f)
    return out
