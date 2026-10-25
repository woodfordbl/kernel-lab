import pytest
import torch
import torch.nn.functional as Fn

from common import DEVICE, DTYPE, ON_GPU
from ex05_matmul_swiglu.kernel import gateup_swiglu

TOL = dict(atol=2e-2, rtol=2e-2) if DTYPE == torch.float16 else dict(atol=1e-4, rtol=1e-4)

SHAPES = [(16, 32, 32), (33, 70, 65)]
if ON_GPU:
    SHAPES += [(256, 512, 1024), (16, 2048, 4096), (1000, 1000, 512)]


@pytest.mark.parametrize("M,F,K", SHAPES)
def test_gateup_swiglu(M, F, K):
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE) / K**0.5
    wg = torch.randn(K, F, device=DEVICE, dtype=DTYPE)
    wu = torch.randn(K, F, device=DEVICE, dtype=DTYPE)
    ref = (Fn.silu(x.float() @ wg.float()) * (x.float() @ wu.float())).to(DTYPE)
    torch.testing.assert_close(gateup_swiglu(x, wg, wu), ref, **TOL)
