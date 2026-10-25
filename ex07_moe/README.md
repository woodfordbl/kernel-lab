# Exercise 7: fused-MoE grouped GEMM

**Goal:** write the core kernel of every MoE serving engine: one launch that runs all
experts, gathers token rows on the fly, and scatters weighted results back.

## Do

1. Read `align.py` and work its example by hand. `uv run poe test 7 -k align` already passes.
2. Fill in `moe_gemm_kernel` (four TODOs). `uv run poe test 7 -k moe_gemm`
3. `moe_mlp` is wired up around your kernel. `uv run poe test 7` runs it end to end.
4. `python -m ex07_moe.bench` in Colab

## Concepts

- **Grouped GEMM.** One grid covers every expert's tiles. Each program reads
  `expert_ids[pid_m]` to find its expert, then offsets into `w` by
  `expert * stride_we`. Experts that got no tokens get no tiles and cost nothing.
- **Gather in the load.** Row r of the A tile is `x[sorted_ids[r] // top_k]`. The pointer
  tile is still `rows[:, None] * stride + cols[None, :]`, but `rows` came from memory
  instead of `tl.arange`. No permuted copy of x is ever written.
- **Padding is wasted work.** Each expert's group is padded to BLOCK_M. In decode, with
  2 or 3 rows per expert, a BLOCK_M = 64 tile is over 95% padding. That's why the bench
  tries 16, 32 and 64.
- **Scatter in the store.** Each output row goes back to its pair's slot, so the final
  combine is a plain reshape and sum over top_k.

## What to look for

- T = 32: memory-bound on expert weights. Compare your GB/s with exercise 1's roof. Small
  BLOCK_M should win.
- T = 4096: about 256 rows per expert, so this behaves like a dense GEMM. Large BLOCK_M wins.
  Production kernels switch configs on exactly this: tokens per expert.
- The torch loop pays one launch, one gather and one scatter per expert. With 64 experts
  that's hundreds of small kernels.

## Stretch

- Fuse SiLU * up into GEMM 1 with two accumulators, as in exercise 5.
- Fuse the combine: in GEMM 2, `tl.atomic_add` each weighted row straight into `out[token]`.
  What does that do to run-to-run determinism?
- Rewrite `moe_align` as Triton kernels: histogram with atomics, then a prefix sum.
