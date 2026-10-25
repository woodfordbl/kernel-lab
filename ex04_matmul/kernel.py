"""Exercise 4: tiled matrix multiply.  C[M, N] = A[M, K] @ B[K, N]

This is the kernel that does most of a transformer's FLOPs. The structure, top to bottom:
  1. map this program to an output tile (pid_m, pid_n), in an L2-friendly order
  2. build pointer tiles for A and B
  3. loop over K: load a tile of A and a tile of B, multiply-accumulate on tensor cores
  4. epilogue: cast the fp32 accumulator and store the C tile
"""
import torch
import triton
import triton.language as tl

from common import ON_GPU


def _configs():
    if not ON_GPU:  # the CPU interpreter is slow: one small config is enough for testing
        return [triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 4})]
    # (BLOCK_M, BLOCK_N, BLOCK_K, stages, warps). Sized to fit a T4's 64 KB of shared
    # memory; bigger GPUs can afford more stages and bigger tiles. Try adding your own.
    cfgs = [(128, 128, 32, 3, 8), (128, 64, 32, 4, 4), (64, 128, 32, 4, 4),
            (64, 64, 32, 4, 4), (32, 64, 64, 4, 4), (16, 128, 64, 3, 4)]
    return [triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k, "GROUP_M": 8},
                          num_stages=s, num_warps=w) for m, n, k, s, w in cfgs]


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # ---- 1. program -> tile mapping --------------------------------------------------
    # The simple mapping is pid_m = pid // num_pid_n, pid_n = pid % num_pid_n (row-major).
    # It is correct, so start with it. Once tests pass, replace it with "grouped" ordering:
    # walk GROUP_M tile-rows at a time, column by column, so programs running at the same
    # moment share A rows and B columns in L2. See README for the picture.
    # TODO 1: set pid_m, pid_n
    pid_m = 0
    pid_n = 0

    # ---- 2. pointer tiles ------------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # TODO 2: a_ptrs: a [BLOCK_M, BLOCK_K] tile of pointers into A (use the strides)
    #         b_ptrs: a [BLOCK_K, BLOCK_N] tile of pointers into B

    # ---- 3. main loop ----------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # TODO 3: for k0 in range(0, K, BLOCK_K):
    #           load the A and B tiles. Mask the rows/cols past M or N AND the K tail.
    #           acc += tl.dot(a, b)       <- tensor cores
    #           advance a_ptrs and b_ptrs by BLOCK_K along K

    # ---- 4. epilogue -----------------------------------------------------------------
    # TODO 4: build c_ptrs and a 2-D mask, store acc cast to C's dtype
    pass


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1))
    return c
