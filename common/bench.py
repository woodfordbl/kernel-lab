"""Timing helper. Uses triton.testing.do_bench, which warms up, flushes L2 between
runs, and returns the median time in milliseconds."""
import torch
import triton


def time_ms(fn, warmup: int = 25, rep: int = 100) -> float:
    if not torch.cuda.is_available():
        raise SystemExit("Benchmarks need a CUDA GPU. Run this in Colab (see colab.ipynb).")
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)
