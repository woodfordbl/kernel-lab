# Exercise 1: vector add

**Goal:** learn the Triton programming model and measure your GPU's real memory bandwidth.

`out = x + y` does 1 FLOP per 6 bytes (fp16: read 2, read 2, write 2). It is as
memory-bound as a kernel can be, so "fast" means "close to peak bandwidth".

## Do

1. Fill in `add_kernel` in `kernel.py` (four TODOs).
2. `uv run poe test 1` until green.
3. `python -m ex01_vector_add.bench` in Colab

## What to look for in the benchmark

- At n = 65k, everyone is slow. The kernel takes a few microseconds and most of that is
  launch overhead. Small kernels can't reach the roof.
- At n = 16M and above, a good kernel reaches 80-90% of peak bandwidth. That number is your
  GPU's practical memory roof; remember it.
- Compare block sizes. Tiny blocks mean too many programs and too few bytes in flight per
  program. Note which size wins.

## Stretch

- Pass `num_warps=` to the launch (`add_kernel[grid](..., num_warps=8)`) and see if it moves.
- Write `add_scaled(x, y, alpha)` computing `alpha * x + y`. Does the time change? Why not?
