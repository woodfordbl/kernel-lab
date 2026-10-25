"""Run from the repo root:  python -m ex05_matmul_swiglu.bench"""
import torch
import torch.nn.functional as Fn

from common import DEVICE, DTYPE, report, time_ms
from ex05_matmul_swiglu.kernel import gateup_swiglu

K, F = 4096, 14336
wg = torch.randn(K, F, device=DEVICE, dtype=DTYPE)
wu = torch.randn(K, F, device=DEVICE, dtype=DTYPE)
w_cat = torch.cat([wg, wu], dim=1)  # how real models store it: one [K, 2F] matrix

for M in [16, 4096]:
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    es = x.element_size()
    flops = 2 * M * K * 2 * F
    fused_bytes = (M * K + 2 * K * F + M * F) * es

    def unfused():
        gu = x @ w_cat                          # writes [M, 2F]
        return Fn.silu(gu[:, :F]) * gu[:, F:]   # reads it back (twice, in eager)

    print(f"M = {M} tokens")
    base = time_ms(unfused)
    report("  torch: GEMM then silu*mul", base, fused_bytes, flops)
    report("  triton fused", time_ms(lambda: gateup_swiglu(x, wg, wu)), fused_bytes, flops, base)
    print()
