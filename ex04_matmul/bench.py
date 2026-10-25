"""Run from the repo root:  python -m ex04_matmul.bench
The first call for each shape autotunes (tries every config), so expect a pause."""
import torch

from common import DEVICE, DTYPE, report, time_ms
from ex04_matmul.kernel import matmul, matmul_kernel

SHAPES = [
    ("square 4096 (compute-bound)", 4096, 4096, 4096),
    ("prefill: 4096 tok x gate/up", 4096, 2 * 14336, 4096),
    ("decode: 16 tok x gate/up", 16, 2 * 14336, 4096),
]

for label, M, N, K in SHAPES:
    a = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    b = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
    es = a.element_size()
    bytes_moved = (M * K + K * N + M * N) * es  # each matrix touched once: the ideal
    flops = 2 * M * N * K
    print(label)
    base = time_ms(lambda: a @ b)
    report("  torch (cuBLAS)", base, bytes_moved, flops)
    report("  triton", time_ms(lambda: matmul(a, b)), bytes_moved, flops, base)
    print(f"  autotune picked: {matmul_kernel.best_config}\n")
