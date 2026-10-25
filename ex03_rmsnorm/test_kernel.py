import pytest
import torch

from common import DEVICE, DTYPE
from ex03_rmsnorm.kernel import add_rmsnorm

TOL = dict(atol=1e-2, rtol=1e-2) if DTYPE == torch.float16 else dict(atol=1e-5, rtol=1e-5)


def ref(x, res, w, eps=1e-6):
    h = x.float() + res.float()
    out = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return out.to(x.dtype), h.to(x.dtype)


# D = 1000 and 4000 are not powers of two: they test your mask and your mean.
@pytest.mark.parametrize("T,D", [(1, 64), (7, 1000), (32, 4096), (3, 4000)])
def test_add_rmsnorm(T, D):
    torch.manual_seed(0)
    x = torch.randn(T, D, device=DEVICE, dtype=DTYPE)
    r = torch.randn(T, D, device=DEVICE, dtype=DTYPE)
    w = torch.randn(D, device=DEVICE, dtype=DTYPE)
    out, h = add_rmsnorm(x, r, w)
    out_ref, h_ref = ref(x, r, w)
    torch.testing.assert_close(h, h_ref, **TOL)
    torch.testing.assert_close(out, out_ref, **TOL)
