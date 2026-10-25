import pytest
import torch

from common import DEVICE, DTYPE
from ex01_vector_add.kernel import add


# Sizes include non-multiples of the block size, to test your mask.
@pytest.mark.parametrize("n", [1, 7, 1024, 1000, 98_432])
@pytest.mark.parametrize("block_size", [128, 1024])
def test_add(n, block_size):
    torch.manual_seed(0)
    x = torch.randn(n, device=DEVICE, dtype=DTYPE)
    y = torch.randn(n, device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(add(x, y, block_size), x + y)
