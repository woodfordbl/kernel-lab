"""Run from the repo root:  python -m ex02_swiglu.bench"""
import torch
import torch.nn.functional as Fn

from common import DEVICE, DTYPE, report, time_ms
from ex02_swiglu.kernel import swiglu, swiglu_packed

# Prefill-sized activations for a Llama-3-8B-like MLP: 4096 tokens x 14336 intermediate.
M, F = 4096, 14336
g = torch.randn(M, F, device=DEVICE, dtype=DTYPE)
u = torch.randn(M, F, device=DEVICE, dtype=DTYPE)
gu = torch.cat([g, u], dim=1)
es = g.element_size()
ideal = 3 * M * F * es   # read gate, read up, write out: the fused minimum

eager = time_ms(lambda: Fn.silu(g) * u)  # two kernels: silu writes a temp, mul re-reads it
report("torch eager silu(g) * u", eager, ideal, 5 * M * F)
print("  (eager actually moves ~5 tensors' worth: read g, write tmp, read tmp, read u, write out)")
report("triton swiglu (part A)", time_ms(lambda: swiglu(g, u)), ideal, 5 * M * F, eager)
report("triton swiglu_packed (part B)", time_ms(lambda: swiglu_packed(gu)), ideal, 5 * M * F, eager)
