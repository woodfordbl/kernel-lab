import pytest
import torch

from common import DEVICE, DTYPE, ON_GPU
from ex06_attention.kernel import flash_attention
from ex06_attention.softmax import softmax

TOL = dict(atol=2e-2, rtol=2e-2) if DTYPE == torch.float16 else dict(atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("R,N", [(1, 3), (5, 1000), (16, 4096)])
def test_softmax(R, N):
    torch.manual_seed(0)
    x = torch.randn(R, N, device=DEVICE, dtype=DTYPE) * 5
    torch.testing.assert_close(softmax(x), torch.softmax(x.float(), -1).to(DTYPE), **TOL)


def ref_attention(q, k, v, causal):
    s = (q.float() @ k.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return (torch.softmax(s, -1) @ v.float()).to(q.dtype)


SHAPES = [(1, 1, 64, 32), (1, 2, 100, 64)]   # N = 100 tests the ragged last block
if ON_GPU:
    SHAPES += [(2, 4, 1024, 64), (1, 8, 777, 128)]


@pytest.mark.parametrize("B,H,N,D", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention(B, H, N, D, causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE) for _ in range(3))
    torch.testing.assert_close(flash_attention(q, k, v, causal), ref_attention(q, k, v, causal), **TOL)
