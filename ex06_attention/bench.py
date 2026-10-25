"""Run from the repo root:  python -m ex06_attention.bench"""
import torch
import torch.nn.functional as Fn

from common import DEVICE, DTYPE, report, time_ms
from ex06_attention.kernel import flash_attention

B, H, D = 1, 16, 64
for N in [1024, 4096]:
    q, k, v = (torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE) for _ in range(3))
    es = q.element_size()
    flops = 4 * B * H * N * N * D / 2          # QK^T and PV, causal half
    io = 4 * B * H * N * D * es                 # read Q, K, V once, write O once

    def naive():
        s = (q @ k.transpose(-1, -2)) * D**-0.5   # [B, H, N, N] written to HBM
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=DEVICE).triu(1), float("-inf"))
        return torch.softmax(s, -1) @ v

    print(f"N = {N}  (naive score matrix: {B*H*N*N*es/1e6:.0f} MB per pass)")
    base = time_ms(naive)
    report("  torch naive (materializes S)", base, io, flops)
    report("  torch SDPA", time_ms(lambda: Fn.scaled_dot_product_attention(q, k, v, is_causal=True)), io, flops, base)
    for bm, bn in [(64, 64), (128, 64), (64, 32)]:
        report(f"  triton flash {bm}x{bn}", time_ms(lambda: flash_attention(q, k, v, True, bm, bn)), io, flops, base)
    print()
