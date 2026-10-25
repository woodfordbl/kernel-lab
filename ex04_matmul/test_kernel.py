import pytest
import torch

from common import DEVICE, DTYPE, ON_GPU
from ex04_matmul.kernel import matmul

TOL = dict(atol=1e-2, rtol=1e-2) if DTYPE == torch.float16 else dict(atol=1e-4, rtol=1e-4)

SHAPES = [(16, 32, 32), (33, 47, 65), (64, 96, 128)]  # ragged shapes test every mask
if ON_GPU:
    SHAPES += [(512, 512, 512), (16, 4096, 1024), (1000, 1000, 1000)]


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_matmul(M, N, K):
    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=DTYPE) / K**0.5
    b = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(matmul(a, b), (a.float() @ b.float()).to(DTYPE), **TOL)


def test_transposed_input():
    # Your kernel uses strides, so it should accept a transposed (non-contiguous) B.
    a = torch.randn(32, 64, device=DEVICE, dtype=DTYPE) / 8
    bt = torch.randn(48, 64, device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(matmul(a, bt.t()), (a.float() @ bt.t().float()).to(DTYPE), **TOL)
