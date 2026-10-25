"""Exercise 5: gate/up GEMM with a fused SwiGLU epilogue.

    out[M, F] = SiLU(x @ W_gate) * (x @ W_up)        x: [M, K], W_gate, W_up: [K, F]

The unfused version runs two GEMMs (or one wide one), writes both [M, F] results to HBM,
then runs exercise 2's kernel to read them back. Here each program keeps TWO accumulators
for the same output tile, loads each x tile once for both, and applies SiLU * up while the
results are still in registers. Only `out` is ever written.
"""
import torch
import triton
import triton.language as tl

from common import ON_GPU


def _configs():
    if not ON_GPU:
        return [triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 4})]
    # Two accumulators double the register use, so these tiles are a step smaller than ex04's.
    cfgs = [(128, 64, 32, 3, 8), (64, 128, 32, 3, 8), (64, 64, 32, 4, 4),
            (32, 64, 64, 3, 4), (16, 64, 64, 3, 4)]
    return [triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k, "GROUP_M": 8},
                          num_stages=s, num_warps=w) for m, n, k, s, w in cfgs]


@triton.autotune(configs=_configs(), key=["M", "F", "K"])
@triton.jit
def gateup_swiglu_kernel(x_ptr, wg_ptr, wu_ptr, out_ptr, M, F, K,
                         stride_xm, stride_xk, stride_wk, stride_wn, stride_om, stride_on,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         GROUP_M: tl.constexpr):
    # Grouped tile ordering, as you wrote in exercise 4.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(F, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    # TODO 1: pointer tiles for W_gate and W_up. Both have the same shape and strides.
    # TODO 2: two fp32 accumulators.
    # TODO 3: K loop: load the x tile ONCE, load both weight tiles, two tl.dot calls.
    # TODO 4: epilogue: out = SiLU(acc_gate) * acc_up, in fp32, then cast and store.
    pass


def gateup_swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    K2, F = w_gate.shape
    assert K == K2 and w_up.shape == w_gate.shape and w_up.stride() == w_gate.stride()
    out = torch.empty((M, F), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(F, meta["BLOCK_N"]),)
    gateup_swiglu_kernel[grid](x, w_gate, w_up, out, M, F, K,
                               x.stride(0), x.stride(1), w_gate.stride(0), w_gate.stride(1),
                               out.stride(0), out.stride(1))
    return out
