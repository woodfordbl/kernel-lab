"""Run from the repo root:  python -m ex01_vector_add.bench"""
import torch

from common import DEVICE, DTYPE, gpu_spec, report, time_ms
from ex01_vector_add.kernel import add

spec = gpu_spec()
print(f"{spec.name}: peak {spec.bandwidth/1e9:.0f} GB/s, ridge {spec.ridge:.0f} FLOP/byte\n")

for n in [2**16, 2**20, 2**24, 2**27]:
    x = torch.randn(n, device=DEVICE, dtype=DTYPE)
    y = torch.randn(n, device=DEVICE, dtype=DTYPE)
    bytes_moved = 3 * n * x.element_size()  # read x, read y, write out
    base = time_ms(lambda: x + y)
    print(f"n = {n:,}")
    report("  torch x + y", base, bytes_moved, n)
    for bs in [128, 512, 1024, 4096]:
        report(f"  triton BLOCK_SIZE={bs}", time_ms(lambda: add(x, y, bs)), bytes_moved, n, base)
    print()
