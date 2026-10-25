# Exercise 3: fused residual add + RMSNorm

**Goal:** write a row reduction, and fuse two layer steps (residual add and the next norm)
into one kernel.

So far every lane of your kernels was independent. Normalization is different: each output
depends on a statistic of the whole row. That needs a **reduction**, the second basic
building block of GPU kernels after the elementwise map.

**Builds on:** exercise 1 (masks, bandwidth roof) and exercise 2 (fp32 math, row strides,
fusion accounting).

## Do

1. Read the Concepts below.
2. Fill in `add_rmsnorm_kernel` (three TODOs).
3. `uv run poe test 3`, then `python -m ex03_rmsnorm.bench` in Colab.

## Concepts

### 1. Where this sits in a transformer layer

A pre-norm transformer layer (Llama and most current models) looks like this:

![Pre-norm transformer layer: the residual stream x runs left to right; a branch through RMSNorm and attention is added back to give h = x + attn(...); h stays on the stream as the new residual and also feeds the next RMSNorm and the MLP. This kernel covers the first add and the second RMSNorm](figures/prenorm-layer.svg)

After attention, its output is added to the residual stream, and the result `h` is
immediately normalized for the MLP. `h` is needed twice: as the new residual (for the next
add) and as the input to RMSNorm. This kernel does both in one pass:

```text
h   = x + residual                         -> written out (the new residual stream)
out = h / sqrt(mean(h^2) + eps) * weight   -> written out (the MLP's input)
```

### 2. What RMSNorm does

RMSNorm rescales each token's hidden vector so its root-mean-square is 1, then applies a
learned per-channel gain `weight` of shape `[D]`:

```text
rms(h) = sqrt( (h_0^2 + h_1^2 + ... + h_{D-1}^2) / D  +  eps )
out_i  = h_i / rms(h) * weight_i
```

It is LayerNorm without the mean subtraction and without the bias. That makes it cheaper and
works as well in practice. `eps` (1e-6 here) keeps the division safe when a row is all zeros.

Each row (token) is normalized independently. Rows never interact, which is what makes
"one program per row" possible.

### 3. One program per row

The grid is `(T,)`: one program per token. Each program:

1. Loads its row of `x` and `residual` (D elements each) and adds them.
2. Stores `h`.
3. Reduces `h * h` to one number, the mean square.
4. Uses that number to scale every element, multiplies by `weight`, and stores `out`.

![One program per row: program row loads its row of x and residual once, keeps h in registers in fp32, stores h, reduces h squared to one mean-square scalar, then scales by rsqrt and weight and stores out. Two rows are read, two are written, nothing is read twice](figures/one-program-per-row.svg)

For D = 4096 in fp16 a row is 8 KB, which fits in the registers of one program's warps
(8 warps x 32 threads x 16 elements each). So the row is loaded **once**. Steps 2 to 4 reuse
it from registers at no memory cost. That is the whole trick: a naive implementation would
read `h` again for the second pass.

Addressing is exercise 2's: row `row` starts at `base + row * stride_row`, and the column
offsets are `tl.arange(0, BLOCK_D)`.

### 4. Reductions on a GPU

`tl.sum(v, axis=0)` adds all the lanes of a block into one value. Under the hood, the lanes
are spread across threads and warps, so the compiler builds a tree:

![Reduction tree for one program of 8 warps x 32 threads x 16 elements: step 0, each thread sums its own 16 elements in registers; step 1, five rounds of warp shuffles with offsets 16, 8, 4, 2 and 1 halve the lanes holding sums until lane 0 holds the warp sum; step 2, each warp writes its sum to shared memory and one warp adds the 8 values; step 3, the row total is broadcast to all 256 threads](figures/reduction-tree.svg)

You don't write any of this. You do need to know it exists: a reduction is cheap, but it
forces threads to wait for each other, and it needs a little shared memory.

`tl.max`, `tl.min` and `tl.sum` all work this way. Exercise 6 uses `tl.max` and `tl.sum` along
`axis=1` of a 2-D block, which reduces each row separately.

**Accumulate in fp32.** Summing 4096 squares in fp16 loses precision fast (fp16 has ~3
significant digits), and large activations can overflow 65504. Load, cast to fp32, then square
and sum.

### 5. Power-of-two blocks and padding lanes

`tl.arange` needs a power-of-two length, so the wrapper sets `BLOCK_D = next_power_of_2(D)`.
For D = 4000, `BLOCK_D = 4096` and 96 lanes are padding.

![Lanes 0 to 3999 are real and masked in; lanes 4000 to 4095 are 96 padding lanes, masked out and filled with the other= value. tl.sum adds all 4096 lanes, padding included, and the mean divides by D = 4000](figures/padding-lanes.svg)

Masked loads don't read memory, but the lanes still exist in the block and still go into
`tl.sum`. So the `other=` value matters now. Ask yourself: what value can padding lanes hold
so they add nothing to a sum of squares?

And the mean divides by the number of **real** elements, D, not by `BLOCK_D`.

This "padding must be the identity of the reduction" rule comes back in every kernel with a
reduction. For a sum the identity is 0; for a max (exercise 6) it is `-inf`.

### 6. Fusion accounting

Count the full-size `[T, D]` transfers.

**The fused kernel's floor:** read `x`, read `residual`, write `h`, write `out` = 4 row-sized
transfers (plus `weight`, which is tiny and stays in cache).

**Eager PyTorch** runs `h = x + r`, `hf = h.float()`, `hf.pow(2)`, `.mean(-1)`, `rsqrt`,
two multiplies and a cast, each as its own kernel. Several of those read and write
**fp32** copies, which are twice the size. A rough count (in fp16-row units):

```text
x + r            read 2, write 1           3
h.float()        read 1, write 2 (fp32)    3
pow(2)           read 2, write 2           4
mean             read 2                    2
hf * rsqrt(...)  read 2, write 2           4
* w              read 2, write 2           4
.to(fp16)        read 2, write 1           3
                                          ~23   vs   4 fused
```

The exact count depends on PyTorch's version and which ops it fuses, but the ratio tells you
why this kernel can be several times faster than eager at large T.

### 7. Occupancy: when one program per row is not enough

At decode time T is small: 32 tokens means 32 programs, on a GPU with 40 SMs (T4) or 132
(H100). Most SMs get no work at all. The kernel finishes in a few microseconds either way,
and launch overhead dominates, but it is your first meeting with an important limit:
**a grid smaller than the number of SMs cannot use the whole GPU**. Exercise 6's decode
stretch and the Flash-Decoding idea in `AGENTS.md` are about the same problem.

## What to look for

- **T = 32 (decode):** a few microseconds. Launch overhead dominates, so "% of roof" will be
  low for everyone. Compare the speedup over eager rather than the roof.
- **T = 4096 and 16384 (prefill):** you should be near your memory roof from exercise 1, and
  several times faster than eager.
- If you're well below the roof at large T, check `num_warps` (the wrapper uses 8 for
  `BLOCK_D > 2048`) and whether your loads are masked and coalesced.

## Check yourself

1. Why must `h` be written to memory even though the kernel already used it?
2. With D = 4000, what goes wrong if you load with no `other=`? What if you divide by
   `BLOCK_D`?
3. How many bytes does the fused kernel move at T = 4096, D = 4096 in fp16? How long should
   that take at your measured bandwidth?
4. What changes in the kernel if this were LayerNorm (subtract the mean, add a bias)? How many
   reductions would you need?
5. At T = 32, how many SMs on your GPU have nothing to do?

## Stretch

- Make it work for D = 32768 without holding the whole row: loop over the row in chunks
  twice (once for the sum of squares, once to normalize). What does the second pass cost?
- Process 2 or 4 rows per program for small D. When does that help?
- Write the LayerNorm version. Can you compute the mean and the variance in one pass over the
  row? (Look up Welford's algorithm, or use `E[h^2] - E[h]^2` and think about its accuracy.)

## Key takeaways

- A row reduction: one program per row, keep the row in registers, `tl.sum` across lanes.
- Padding lanes must hold the identity of the reduction (0 for sum). Divide by the real
  length, not the block length.
- Accumulate reductions in fp32.
- Fusing the residual add with the norm saves both launches and several fp32 round trips;
  the floor is 4 row-sized transfers.
- A grid smaller than the number of SMs leaves the GPU idle; decode shapes hit this.
