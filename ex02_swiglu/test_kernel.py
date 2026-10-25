import pytest
import torch
import torch.nn.functional as Fn

from common import DEVICE, DTYPE
from ex02_swiglu.kernel import swiglu, swiglu_packed

TOL = dict(atol=2e-3, rtol=2e-3) if DTYPE == torch.float16 else dict(atol=1e-5, rtol=1e-5)


def ref(g, u):
    return (Fn.silu(g.float()) * u.float()).to(g.dtype)


@pytest.mark.parametrize("shape", [(1, 5), (4, 1000), (64, 1536)])
def test_swiglu(shape):
    torch.manual_seed(0)
    g = torch.randn(shape, device=DEVICE, dtype=DTYPE)
    u = torch.randn(shape, device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(swiglu(g, u), ref(g, u), **TOL)


@pytest.mark.parametrize("M,F", [(1, 7), (33, 200), (64, 512)])
def test_swiglu_packed(M, F):
    torch.manual_seed(0)
    gu = torch.randn(M, 2 * F, device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(swiglu_packed(gu), ref(gu[:, :F], gu[:, F:]), **TOL)
