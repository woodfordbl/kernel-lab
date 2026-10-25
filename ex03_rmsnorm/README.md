# Exercise 3: fused residual add + RMSNorm

**Goal:** write a row reduction, and fuse two layer steps (residual add and the next norm)
into one kernel. This is steps 6 + 7 of the Layer Trace.

## Do

1. Fill in `add_rmsnorm_kernel` (three TODOs).
2. `uv run poe test 3`, then `python -m ex03_rmsnorm.bench` in Colab

## Concepts

- **One program per row.** D = 4096 fp16 values is 8 KB, small enough to hold in registers
  across the program's warps. Load the row once, and every later step reuses it for free.
- **Reductions.** `tl.sum(v, axis=0)` sums a vector across all lanes of the program.
  Triton turns it into warp shuffles plus a small shared-memory exchange between warps.
- **Masked lanes.** `BLOCK_D` is rounded up to a power of 2, so for D = 4000, 96 lanes are
  padding. `tl.load(..., mask=mask, other=0.0)` makes them zeros, so they add nothing to the
  sum. But the mean must still divide by D.
- **Fusion accounting.** Eager PyTorch writes and re-reads `h`, `h.float()`, `h^2`, and more.
  The fused kernel's floor is 4 row-sized transfers. Count them in `bench.py`.

## What to look for

At T = 32 (decode) the kernel is a few microseconds and launch overhead dominates. At
T = 4096+ you should be near your memory roof from exercise 1.

## Stretch

- Make it work for D = 32768 without holding the whole row: loop over the row in chunks
  twice (once for the sum of squares, once to normalize). What does the second pass cost?
- Process 2 or 4 rows per program for small D. When does that help?
