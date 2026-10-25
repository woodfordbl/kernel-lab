# Exercise 2: fused SwiGLU

**Goal:** see kernel fusion pay off on a memory-bound op, and learn 2-D tile indexing.

PyTorch eager runs `F.silu(g) * u` as two kernels. The first writes `silu(g)` to HBM and
the second reads it back. A fused kernel reads `g` and `u` once and writes `out` once:
3 tensors of traffic instead of 5, so you should approach a **1.6x speedup** without
touching the math.

**Builds on:** exercise 1 (the elementwise kernel skeleton, masks, and your bandwidth roof).

## Do

1. Read Concepts 1 to 3, then do Part A: `swiglu_kernel`, 1-D, the same skeleton as
  exercise 1 plus fp32 math.
2. Read Concepts 4 to 7, then do Part B: `swiglu_packed_kernel`, 2-D tiles over a packed
  `[M, 2F]` input.
3. `uv run poe test 2`, then `python -m ex02_swiglu.bench` in Colab.



## Concepts



### 1. Where SwiGLU sits in a transformer

Each transformer layer has an MLP block. Modern models (Llama, Mistral, Qwen) use a
gated MLP:

![The gated MLP: x [M, D] goes through two GEMMs, x @ W_gate and x @ W_up, giving gate and up, each [M, F]. SwiGLU, the part this exercise writes, applies SiLU to gate and multiplies by up to give h [M, F], and h @ W_down gives the [M, D] output](figures/gated-mlp.svg)

`M` is the number of tokens, `D` the model width (4096 for Llama-3-8B) and `F` the MLP
width (14336). The activation in the middle is

```text
SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
SwiGLU(g, u) = SiLU(g) * u
```

The two GEMMs do almost all the FLOPs. The activation does about 5 FLOPs per element, which
is nothing, but it touches three `[M, F]` tensors. At M = 4096 and F = 14336 each one is
117 MB in fp16. That makes the activation a pure bandwidth problem, like vector add.

### 2. Fusion: count tensors, not FLOPs

For a memory-bound op, time ≈ bytes moved / bandwidth. So count the full-size tensors each
version reads or writes.

![Eager PyTorch runs two kernels: kernel 1 reads g and writes tmp to HBM, kernel 2 reads tmp and u and writes out, 5 tensors of traffic. The fused kernel reads g and u and writes out, keeping silu(g) in registers, 3 tensors of traffic](figures/fusion-traffic.svg)

Best-case speedup: 5 / 3 ≈ 1.67x. On a T4 at ~280 GB/s practical bandwidth, 3 x 117 MB
takes about 1.25 ms and 5 x 117 MB about 2.1 ms.

This is the core idea of the whole repo: **the intermediate never touches global memory**.
`silu(g)` exists only in registers, between the load and the store. Exercises 3, 5, 6 and 7
apply the same idea to larger pieces of the layer.

### 3. Mixed precision: load low, compute high, store low

Tensors are stored in fp16 (or bf16) to halve memory traffic, but fp16 has only a 10-bit
mantissa (about 3 decimal digits) and a maximum value of 65504. The usual rule:

1. Load in the storage dtype.
2. Cast to fp32: `.to(tl.float32)`.
3. Do the math in fp32.
4. Cast back to the output dtype before the store.

The casts cost almost nothing here. The kernel spends its time waiting for memory, and the
SM has plenty of idle ALU cycles to do fp32 math and conversions while it waits. In a
memory-bound kernel, extra math is free until you get close to the ridge point.

Note: on the CPU interpreter the tests run in fp32, so a missing cast won't fail there. It
will show up as a larger error on the GPU.

### 4. A 2-D tensor is a 1-D buffer plus strides

Memory is flat. A row-major `[M, N]` tensor stores row 0, then row 1, and so on. The
**stride** of a dimension is how many elements you skip to move one step along it.

![A contiguous 3 x 4 tensor t holding a to l is stored row after row in one flat buffer at indices 0 to 11, so t.stride() is (4, 1). The view t[:, 1:3] picks b, c, f, g, j, k at indices 1, 2, 5, 6, 9, 10: the same buffer with base offset 1, shape (3, 2) and strides (4, 1)](figures/strides-view.svg)

```text
element [r, c] lives at  base + r * stride_row + c * stride_col
```

A slice like `t[:, 1:3]` is a **view**: same buffer, same strides, different base pointer
and shape. Its rows are no longer next to each other in memory. A kernel that is passed
strides, instead of assuming `stride_row == N`, works on views too.

In this exercise the column stride is 1, so the kernel only takes row strides.

### 5. The packed layout

Real models often store `W_gate` and `W_up` side by side as one `[D, 2F]` matrix, so a
single GEMM produces both. Its output is one `[M, 2F]` tensor `gu`:

![gu is [M, 2F]: in each row, columns 0 to F-1 are gate (gu[:, :F]) and columns F to 2F-1 are up (gu[:, F:]). In memory the rows follow each other, each holding its gate half then its up half, so stride_gu_row = 2F. The output out is [M, F] with stride_out_row = F. Example with M = 3, F = 4, with gate[1, 1] and out[1, 1] highlighted](figures/packed-layout.svg)

Part B reads this layout directly. Work out on paper: for row `r` and column `c < F`, what
is the flat offset of `gate[r, c]`? Of `up[r, c]`? Of `out[r, c]` in the `[M, F]` output?

### 6. 2-D tiles and broadcasting

In Part B the grid is 2-D: `grid = (cdiv(M, BLOCK_M), cdiv(F, BLOCK_F))`, and program
`(pid_m, pid_f)` owns one `BLOCK_M x BLOCK_F` tile of the output.

```text
               F columns of out
          +--------+--------+--------+
 M rows   | (0,0)  | (0,1)  | (0,2)  |
          +--------+--------+--------+
          | (1,0)  | (1,1)  | (1,2)  |   each box is one program's tile
          +--------+--------+--------+
```

Each program builds a vector of row indices (length `BLOCK_M`) and a vector of column
indices (length `BLOCK_F`). **Broadcasting** turns two 1-D vectors into a 2-D block:

```text
rows = [10, 11, 12]          rows[:, None]  is a column  (3 x 1)
cols = [ 0,  1,  2,  3]      cols[None, :]  is a row     (1 x 4)

rows[:, None] * S + cols[None, :]  =  [[10S+0, 10S+1, 10S+2, 10S+3],
                                       [11S+0, 11S+1, 11S+2, 11S+3],
                                       [12S+0, 12S+1, 12S+2, 12S+3]]
```

That is a whole tile of offsets in one expression. Add it to a base pointer and you have a
tile of pointers. Every GEMM in exercises 4 to 7 builds its tiles this way.

The mask is 2-D too: a lane is valid only if its row is in range **and** its column is in
range. Build a row mask and a column mask, broadcast them the same way, and combine with `&`.

### 7. Coalescing: why the column index goes last

When the 32 threads of a warp load 32 consecutive addresses, the hardware merges them into a
few wide memory transactions. This is **coalescing**, and it is how you get full bandwidth.
If they load 32 scattered addresses, each needs its own transaction and most of the bytes
fetched are thrown away.

In a row-major tensor, consecutive columns are consecutive addresses. Triton lays out a
tile so that the last dimension is the fastest-varying one across threads, which is why the
offsets put rows on the first axis and columns (stride 1) on the second. Keep `BLOCK_F`
reasonably large (128 by default) so each row segment is long and contiguous.

## What to look for

- **Part A vs eager:** close to 1.6x. If you're well below that, check the GB/s column
against your exercise 1 roof. A fused kernel should be near it.
- **Part B vs Part A:** similar speed. They move the same bytes; only the addressing differs.
If Part B is much slower, suspect uncoalesced access (rows and columns swapped).
- **The torch line's GB/s** is computed from the ideal 3-tensor count, so it understates the
bandwidth eager really uses. Its "% of roof" is really "% of what a fused kernel could do".



## Check yourself

1. Eager PyTorch launches two kernels here. How many bytes does it move at M = 4096,
  F = 14336 in fp16? How many does your fused kernel move?
2. Why is computing `sigmoid` in fp32 free in this kernel but would not be free in a
  compute-bound kernel?
3. For `t = torch.empty(8, 6)`, what are `t.stride()` and `t[:, 2:5].stride()`? What is the
  base offset of the view?
4. In Part B, if F = 100 and `BLOCK_F = 128`, how many columns of the last tile are masked?
  What happens if you forget the column mask? (Hint: what is just to the right of column
   F - 1 in `gu`?)
5. Why does Part B take `stride_gu_row` and `stride_out_row` separately?



## Stretch

- Part B is the epilogue you'll fuse into a GEMM in exercise 5. Estimate the bytes saved
when `gu` never reaches HBM at all.
- Try `BLOCK_M = 1, BLOCK_F = 1024` and `BLOCK_M = 64, BLOCK_F = 16` in Part B. Which is
slower and why?
- Make Part A handle bf16 inputs. Does anything in your kernel change?



## Key takeaways

- For memory-bound ops, speedup from fusion ≈ (tensors moved before) / (tensors moved after).
- Load in storage precision, compute in fp32, store in storage precision.
- `address = base + row * stride_row + col * stride_col`. Pass strides; don't assume contiguity.
- `rows[:, None] * stride + cols[None, :]` builds a 2-D tile of pointers. 2-D masks combine
a row mask and a column mask with `&`.
- Contiguous addresses across a warp (coalescing) are what make bandwidth possible.

