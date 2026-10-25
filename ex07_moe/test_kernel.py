import pytest
import torch
import torch.nn.functional as Fn

from common import DEVICE, DTYPE, ON_GPU
from ex07_moe.align import moe_align
from ex07_moe.kernel import moe_gemm, moe_mlp

TOL = dict(atol=3e-2, rtol=3e-2) if DTYPE == torch.float16 else dict(atol=1e-4, rtol=1e-4)


def test_align_example():
    ids = torch.tensor([[0, 2], [2, 1], [0, 2]], device=DEVICE)
    s, e = moe_align(ids, 3, 4)
    assert s.tolist() == [0, 4, 6, 6, 3, 6, 6, 6, 1, 2, 5, 6]
    assert e.tolist() == [0, 1, 2]


def ref_gemm(x, w, topk_ids, topk_w, mul):
    T, k = topk_ids.shape
    out = torch.empty(T * k, w.shape[2], device=x.device, dtype=torch.float32)
    for p in range(T * k):
        e = int(topk_ids.flatten()[p])
        out[p] = x[p // k].float() @ w[e].float()
        if mul:
            out[p] *= topk_w.flatten()[p].float()
    return out.to(x.dtype)


CASES = [(5, 4, 2, 32, 48, 16), (9, 8, 3, 64, 40, 16)]  # T, E, top_k, K, N, block_m
if ON_GPU:
    CASES += [(64, 16, 4, 256, 128, 64), (33, 64, 6, 512, 96, 16)]


@pytest.mark.parametrize("T,E,k,K,N,bm", CASES)
@pytest.mark.parametrize("mul", [False, True])
def test_moe_gemm(T, E, k, K, N, bm, mul):
    torch.manual_seed(0)
    x = torch.randn(T, K, device=DEVICE, dtype=DTYPE) / K**0.5
    w = torch.randn(E, K, N, device=DEVICE, dtype=DTYPE)
    topk_ids = torch.stack([torch.randperm(E, device=DEVICE)[:k] for _ in range(T)])
    topk_w = torch.rand(T, k, device=DEVICE, dtype=DTYPE)
    sorted_ids, expert_ids = moe_align(topk_ids, E, bm)
    out = moe_gemm(x, w, topk_ids, topk_w, sorted_ids, expert_ids, mul, bm)
    torch.testing.assert_close(out, ref_gemm(x, w, topk_ids, topk_w, mul), **TOL)


def ref_mlp(x, w13, w2, logits, k):
    probs = torch.softmax(logits.float(), -1)
    tw, ti = torch.topk(probs, k, -1)
    tw = tw / tw.sum(-1, keepdim=True)
    F = w13.shape[2] // 2
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for t in range(x.shape[0]):
        for j in range(k):
            e = int(ti[t, j])
            h = x[t].float() @ w13[e].float()
            out[t] += tw[t, j] * ((Fn.silu(h[:F]) * h[F:]) @ w2[e].float())
    return out.to(x.dtype)


def test_moe_mlp():
    torch.manual_seed(0)
    T, D, F, E, k = (12, 64, 32, 8, 2) if not ON_GPU else (48, 256, 128, 16, 4)
    x = torch.randn(T, D, device=DEVICE, dtype=DTYPE)
    w13 = torch.randn(E, D, 2 * F, device=DEVICE, dtype=DTYPE) / D**0.5
    w2 = torch.randn(E, F, D, device=DEVICE, dtype=DTYPE) / F**0.5
    logits = torch.randn(T, E, device=DEVICE, dtype=DTYPE)
    bm = 16
    torch.testing.assert_close(moe_mlp(x, w13, w2, logits, k, bm), ref_mlp(x, w13, w2, logits, k), **TOL)
