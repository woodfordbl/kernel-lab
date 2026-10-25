# Exercise 2: fused SwiGLU

**Goal:** see kernel fusion pay off on a memory-bound op, and learn 2-D tile indexing.

PyTorch eager runs `F.silu(g) * u` as two kernels. The first writes `silu(g)` to HBM and
the second reads it back. A fused kernel reads `g` and `u` once and writes `out` once:
3 tensors of traffic instead of 5, so you should approach a **1.6x speedup** without
touching the math.

## Do

1. Part A: `swiglu_kernel`, 1-D, same skeleton as exercise 1 plus fp32 math.
2. Part B: `swiglu_packed_kernel`, 2-D tiles over a packed `[M, 2F]` input.
3. `uv run poe test 2`, then `python -m ex02_swiglu.bench` in Colab

## Concepts

- **Upcasting.** `sigmoid` in fp16 loses precision. Load fp16, compute in fp32, store fp16.
  It costs nothing, because the kernel is waiting on memory anyway.
- **Strides.** A 2-D tensor is a flat buffer. Element `[r, c]` lives at
  `base + r * stride_row + c`. Pass strides in instead of assuming contiguity, and your
  kernel works on views and slices too.
- **Broadcasting pointers.** `rows[:, None] * stride + cols[None, :]` builds a whole
  BLOCK_M x BLOCK_F tile of addresses in one line. Every GEMM in exercises 4-7 uses this.

## Stretch

- Part B is the epilogue you'll fuse into a GEMM in exercise 5. Estimate the bytes saved
  when `gu` never reaches HBM at all.
