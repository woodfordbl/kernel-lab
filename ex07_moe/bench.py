"""Run from the repo root:  python -m ex07_moe.bench

Same example config as the Layer Trace: D=4096, 64 experts, top-4, expert FFN 2048.
Weights are ~3.2 GB in fp16, which fits on a T4's 16 GB."""
import torch
import torch.nn.functional as Fn

from common import DEVICE, DTYPE, report, time_ms
from ex07_moe.kernel import moe_mlp

D, F, E, K_TOP = 4096, 2048, 64, 4
w13 = torch.randn(E, D, 2 * F, device=DEVICE, dtype=DTYPE) / D**0.5
w2 = torch.randn(E, F, D, device=DEVICE, dtype=DTYPE) / F**0.5
es = w13.element_size()


def torch_loop(x, logits):
    """The obvious PyTorch version: route, then loop over experts with a gather each."""
    probs = torch.softmax(logits.float(), -1)
    tw, ti = torch.topk(probs, K_TOP, -1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(x.dtype)
    out = torch.zeros_like(x)
    for e in range(E):
        tok, slot = torch.where(ti == e)
        if tok.numel() == 0:
            continue
        h = x[tok] @ w13[e]
        y = (Fn.silu(h[:, :F]) * h[:, F:]) @ w2[e]
        out.index_add_(0, tok, y * tw[tok, slot, None])
    return out


for T in [32, 4096]:
    x = torch.randn(T, D, device=DEVICE, dtype=DTYPE)
    logits = torch.randn(T, E, device=DEVICE, dtype=DTYPE)
    ti = torch.topk(logits, K_TOP, -1).indices
    touched = ti.unique().numel()
    weight_bytes = touched * 3 * D * F * es
    act_bytes = (T * D + T * K_TOP * (2 * F + F + D) * 2 + T * D) * es
    flops = 2 * T * K_TOP * 3 * D * F
    print(f"T = {T}: {touched}/{E} experts touched, {T * K_TOP / touched:.1f} rows per expert")
    base = time_ms(lambda: torch_loop(x, logits))
    report("  torch per-expert loop", base, weight_bytes + act_bytes, flops)
    for bm in [16, 32, 64]:
        report(f"  triton moe_mlp BLOCK_M={bm}", time_ms(lambda: moe_mlp(x, w13, w2, logits, K_TOP, bm)),
               weight_bytes + act_bytes, flops, base)
    print()
