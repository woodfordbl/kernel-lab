# Exercise 4: tiled matmul

**Goal:** build a tiled GEMM. Get within ~80% of cuBLAS on large shapes, then see why the
decode shape is a different problem.

Matrix multiply does most of a transformer's FLOPs: the Q/K/V/O projections, the MLP's
gate/up/down projections and the LM head are all GEMMs. It is also the first kernel in this
repo that can be **compute-bound**, so the question changes from "how close to peak
bandwidth?" to "how close to peak tensor-core throughput?".

**Builds on:** exercise 2 (2-D tiles of pointers, strides, 2-D masks) and exercise 1 (the
roofline).

## Do

1. Read Concepts 1 to 5.
2. Do TODO 1 with the simple row-major mapping, then TODOs 2 to 4. `uv run poe test 4`
   until green.
3. `python -m ex04_matmul.bench` in Colab and write down the numbers.
4. Read Concepts 6 and 7. Replace TODO 1 with grouped ordering. Re-run the bench and compare.

## Concepts

### 1. The math and its cost

`C[M, N] = A[M, K] @ B[K, N]`, where each output is a dot product of length K:

```text
C[i, j] = sum over k of A[i, k] * B[k, j]
```

- FLOPs: `2 * M * N * K` (one multiply and one add per term).
- Minimum bytes: read A and B once, write C once: `(M*K + K*N + M*N) * 2` in fp16.

```text
shape                    FLOPs        min bytes    AI (FLOP/byte)
square 4096^3            137 GFLOP    100 MB       ~1365          compute-bound
prefill 4096 x 28672     962 GFLOP    500 MB       ~1900          compute-bound
  x 4096
decode 16 x 28672        3.8 GFLOP    236 MB       ~16            memory-bound
  x 4096
```

The ridge on a T4 is ~203 FLOP/byte. So the same kernel faces two completely different
problems depending on M: large M is limited by tensor cores, small M by reading the weight
matrix B.

### 2. Why tiling: reuse

A naive kernel computes each `C[i, j]` by reading a full row of A and a full column of B
from global memory. Every element of A gets read N times and every element of B M times.
The arithmetic intensity is about 0.5 FLOP/byte, no better than vector add.

**Tiling** fixes this. Each program owns a `BLOCK_M x BLOCK_N` tile of C. It walks along K
in steps of `BLOCK_K`, loading one tile of A and one tile of B per step:

![One program's K loop: at each step it loads one A tile from its row strip and one B tile from its column strip, and adds their product to an fp32 accumulator for its C tile](figures/tiling.svg)

Each A tile, once loaded, is used against all `BLOCK_N` columns; each B tile against all
`BLOCK_M` rows. Per K step the program loads `(BLOCK_M + BLOCK_N) * BLOCK_K` elements and does
`2 * BLOCK_M * BLOCK_N * BLOCK_K` FLOPs:

```text
tile AI (fp16) = BLOCK_M * BLOCK_N / (BLOCK_M + BLOCK_N)   FLOP/byte

 32 x  32  ->  16
 64 x  64  ->  32
128 x 128  ->  64
```

Bigger tiles mean more reuse. Note that even 128 x 128 (64 FLOP/byte) is below the T4's ridge
of ~203. Tiles alone don't make the GEMM compute-bound from DRAM's point of view; the rest
comes from the **L2 cache** serving repeated tile loads (Concept 7).

### 3. The accumulator and the K tail

The accumulator `acc` is a `BLOCK_M x BLOCK_N` block of fp32 values, held in registers for
the whole K loop. It starts at zero; each step adds one tile product. Only after the loop is
it cast to fp16 and stored (the **epilogue**). Accumulating in fp32 matters: a dot product of
length 4096 summed in fp16 would lose most of its precision.

When K is not a multiple of `BLOCK_K`, the last step reads past the end of the K dimension.
Those lanes must be masked, and since they feed into a dot product, the padding must be 0
(the identity of addition, as in exercise 3). The amount of K left at step `k0` is `K - k0`.
So there are three masks in play: rows of A past M, columns of B past N, and the K tail on
both.

Moving to the next K step is pointer arithmetic: A advances along its columns, B along its
rows, each by `BLOCK_K` times the right stride.

### 4. `tl.dot` and tensor cores

`tl.dot(a, b)` multiplies two blocks and lowers to **tensor-core** instructions: `mma` on
T4/A100, `wgmma` on H100. A tensor core multiplies small matrix fragments (e.g. 16x8x8 on
Turing) in one instruction. On a T4, fp16 tensor cores deliver 65 TFLOP/s versus about
8 TFLOP/s for ordinary fp32 math, an 8x difference.

Rules:

- Inputs fp16 or bf16 (or fp8/int8 on newer GPUs), accumulator fp32.
- Every tile dimension must be at least 16.
- Shapes must line up: `[BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]`. Build the B pointer tile
  with K on the first axis.

### 5. The program-to-tile mapping

The grid is 1-D: `cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)` programs, one per C tile. Each program
must turn its `pid` into `(pid_m, pid_n)`. The simplest choice is row-major:

```text
pid_m = pid // num_pid_n
pid_n = pid %  num_pid_n
```

It's correct, so start with it. Concept 7 explains why it's slow.

### 6. Pipelining, `num_stages` and `num_warps`

A global load takes hundreds of cycles. If the program loads a tile and then waits for it
before computing, the tensor cores sit idle most of the time.

**Software pipelining** overlaps the two. With `num_stages = 3`, while the tensor cores work
on step k, the loads for steps k+1 and k+2 are already in flight into shared memory:

![Timeline: with num_stages = 1, loads and compute alternate and the tensor cores idle during every load; with num_stages = 3, the next tiles load while the current one computes, and the loop finishes much sooner](figures/pipeline.svg)

Triton does this automatically for a `for` loop over K. You choose the depth. The cost is
shared memory: each stage holds one A tile and one B tile.

```text
128 x 128 x 32 tile, fp16:  (128*32 + 32*128) * 2 B = 16 KB per stage
3 stages = 48 KB        fits a T4's 64 KB of shared memory
4 stages = 64 KB        at the limit (the compiler also needs some)
```

`num_warps` sets how many warps share a program's work: more warps give more threads to
issue loads and tensor-core instructions, but each gets fewer registers.

### 7. Grouped ordering: making L2 work for you

Programs run in waves: a T4 runs a few programs per SM at once across 40 SMs, so roughly the
first hundred-odd `pid`s run together, then the next batch, and so on. Which tiles run at the
same time decides how much they share in L2 (4 MB on a T4).

With **row-major** order, concurrent programs all sit in one row of C tiles. They share the
same A rows, but each needs a different B column strip, so most of B is fetched from DRAM
again by every row of programs.

With **grouped** order, consecutive `pid`s walk down a group of `GROUP_M` tile-rows before
moving one column right. Concurrent programs then cover a squarer block of C and share both
A rows and B columns in L2:

![Two 4x4 grids of C tiles numbered by program id. If programs 0 to 3 run together, row-major order needs 1 A strip and 4 B strips; grouped order with GROUP_M = 2 needs 2 A strips and 2 B strips](figures/grouped-order.svg)

In this toy case grouping loads 4 strips instead of 5 for the same work. For real shapes,
with ~100 concurrent programs, the gap is much larger: about 2√P strips instead of P + 1 for
P concurrent programs.

The formula, in words:

- A **group** is `GROUP_M` tile-rows times all `num_pid_n` columns, so it holds
  `GROUP_M * num_pid_n` programs. `group = pid // (GROUP_M * num_pid_n)`.
- The group's first tile-row is `group * GROUP_M`.
- The last group may be shorter, because M may not divide evenly: its height is the smaller
  of `GROUP_M` and the number of tile-rows left.
- Within the group, `pid_m` steps fastest (down the column), then `pid_n`.

Derive the two lines for `pid_m` and `pid_n` on paper, and check them against the picture.

### 8. Autotuning

The best `(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)` depends on the shape and the GPU.
`@triton.autotune(configs=..., key=["M", "N", "K"])` benchmarks every config the first time
it sees a new `(M, N, K)` and caches the winner. That's why the first call per shape pauses.
The bench prints the winner; read it and ask why.

On the CPU interpreter there is only one small config, because running six configs through
the interpreter would take far too long.

### 9. Waves and tile quantization

A GEMM with 1024 tiles on 40 SMs runs in about 25 waves, and the last wave being partial
barely matters. A GEMM with 50 tiles runs in 2 waves, the second mostly empty: up to ~40%
of the GPU sits idle. Small or oddly-shaped problems lose to this **wave quantization**, which
is why smaller tiles or split-K (see Stretch) sometimes win even though they reuse less.

## What to look for

- **Square 4096:** compute-bound. 70-80% of cuBLAS is a solid first result. Check the
  TFLOPS column against your GPU's peak.
- **Prefill:** also compute-bound; similar to square.
- **Grouped vs row-major:** expect a visible gain on the large shapes, where B no longer fits
  in L2. Write both numbers down.
- **Decode, 16 tokens:** intensity is about 16, far below the ridge. Look at GB/s, not TFLOPS,
  and compare with your exercise 1 roof. The weight matrix read dominates. Which config did
  autotune pick, and why a small `BLOCK_M`? (How many of a 128-row tile's rows would be real?)

## Check yourself

1. What is the tile arithmetic intensity of a 64 x 128 tile? Of 16 x 128?
2. For the decode shape with `BLOCK_M = 16, BLOCK_N = 128`, how many programs run? How many
   waves is that on a T4, if each SM runs one program at a time?
3. Why must the K-tail padding be 0? What would happen with garbage values there?
4. How much shared memory do 4 stages of a 64 x 128 x 32 tile need?
5. In grouped order with `GROUP_M = 8`, M = 1000 and `BLOCK_M = 128`, how many tile-rows does
   the last group have?

## Stretch

- Add a config with `BLOCK_K = 64` and see what autotune does with it.
- Split-K for the decode shape: split the K loop across several programs and add their
  partial results with `tl.atomic_add`. Does it beat one program per tile? Why might it?
- Make the kernel handle a transposed B (`b.t()` of a contiguous `[N, K]` matrix) with no
  copy. What changes? (Hint: nothing in the kernel, if you used the strides.)

## Key takeaways

- GEMM FLOPs = 2MNK. Large M is compute-bound; small M (decode) is bound by reading weights.
- Tiling gives reuse: tile AI = `BLOCK_M * BLOCK_N / (BLOCK_M + BLOCK_N)` in fp16.
- Accumulate in fp32 registers across the K loop; mask the K tail with zeros; cast and store
  in the epilogue.
- `tl.dot` uses tensor cores (fp16 in, fp32 accumulate, dims ≥ 16).
- `num_stages` overlaps loads with compute at the cost of shared memory.
- Grouped tile ordering makes concurrent programs share A and B in L2.
- Autotune picks the config per shape; wave quantization punishes small grids.
