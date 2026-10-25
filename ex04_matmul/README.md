# Exercise 4: tiled matmul

**Goal:** build the GEMM from Part 2 of the lesson. Get within ~80% of cuBLAS on large
shapes, then see why the decode shape is a different problem.

## Do

1. TODO 1 with the simple row-major mapping. TODOs 2-4. `uv run poe test 4` until green.
2. `python -m ex04_matmul.bench` in Colab and write down the numbers.
3. Replace TODO 1 with grouped ordering. Re-run the bench and compare.

## Concepts

- **Tiles.** Each program owns a BLOCK_M x BLOCK_N tile of C and walks K in BLOCK_K steps.
  Each loaded A tile is reused BLOCK_N times and each B tile BLOCK_M times. That reuse is
  where the arithmetic intensity comes from.
- **`tl.dot`.** Lowers to tensor-core instructions (`mma` on T4/A100, `wgmma` on H100).
  Inputs fp16, accumulator fp32. Tile sizes must be at least 16 in every dimension.
- **`num_stages`.** Triton software-pipelines the K loop: while stage k computes, stages k+1
  and k+2 are already loading. More stages hide more latency but use more shared memory.
- **Autotuning.** The best tile shape depends on the problem shape and the GPU. Autotune tries
  each config once per (M, N, K) and caches the winner.
- **Grouped ordering.** With row-major order, the programs running at the same moment all
  share one row of A tiles but need many different B columns, so B gets re-read from HBM.
  Grouping GROUP_M rows together makes concurrent programs share both:

```
row-major order          grouped (GROUP_M = 2)
 0  1  2  3               0  2  4  6
 4  5  6  7               1  3  5  7
 8  9 10 11               8 10 12 14
12 13 14 15               9 11 13 15
```

The formula: `group = pid // (GROUP_M * num_pid_n)`; first row of the group is
`group * GROUP_M`; the group may be shorter at the bottom edge; within the group,
`pid_m` steps fastest.

## What to look for

- Square 4096: compute-bound. 70-80% of cuBLAS is a solid first result.
- Decode, 16 tokens: intensity is about 16, far below the ridge. Look at GB/s, not TFLOPS.
  The weight matrix read dominates. Which config did autotune pick, and why a small BLOCK_M?

## Stretch

- Add a config with BLOCK_K=64 and see what autotune does with it.
- Split-K for the decode shape: split the K loop across several programs and add their
  partial results with `tl.atomic_add`. Does it beat one program per tile? Why might it?
