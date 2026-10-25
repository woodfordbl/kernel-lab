"""Run from the repo root:  python -m ex03_rmsnorm.bench"""
import torch

from common import DEVICE, DTYPE, report, time_ms
from ex03_rmsnorm.kernel import add_rmsnorm


def torch_eager(x, r, w, eps=1e-6):
    h = x + r
    hf = h.float()
    return (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype), h


D = 4096
for T in [32, 4096, 16384]:  # decode batch, prefill, big prefill
    x = torch.randn(T, D, device=DEVICE, dtype=DTYPE)
    r = torch.randn(T, D, device=DEVICE, dtype=DTYPE)
    w = torch.randn(D, device=DEVICE, dtype=DTYPE)
    es = x.element_size()
    ideal = (4 * T * D + D) * es  # read x, read r, write out, write h, read w
    base = time_ms(lambda: torch_eager(x, r, w))
    print(f"T = {T}")
    report("  torch eager (~5 kernels)", base, ideal, 6 * T * D)
    report("  triton fused add + rmsnorm", time_ms(lambda: add_rmsnorm(x, r, w)), ideal, 6 * T * D, base)
    print()
