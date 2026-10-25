# Exercise 1: vector add

**Goal:** learn the Triton programming model and measure your GPU's real memory bandwidth.

`out = x + y` is the "hello world" of GPU kernels. The math is trivial on purpose: with
nothing interesting to compute, everything you see in the benchmark is the cost of moving
data and launching work. You'll leave with two things you use in every later exercise: the
shape of a Triton kernel (program id, offsets, mask, load, compute, store) and one number,
your GPU's practical memory bandwidth.

**Builds on:** nothing. This is the starting point.

## Do

1. Read the Concepts below, at least sections 1 to 4.
2. Fill in `add_kernel` in `kernel.py` (four TODOs).
3. `uv run poe test 1` until green.
4. `python -m ex01_vector_add.bench` in Colab. Paste the output somewhere you can find it
   again; you'll compare against it in exercises 2, 3 and 7.

## Concepts

### 1. The GPU in one page

A GPU is many small processors, called **streaming multiprocessors (SMs)**, that share one
large, slow memory. A T4 has 40 SMs; an H100 has 132.

![A T4 chip with 40 SMs, each holding its own registers and shared memory, above a 4 MB L2 cache shared by all SMs; below the chip, 16 GB of global memory connected at about 320 GB/s](figures/memory-hierarchy.svg)

- **Global memory (HBM or GDDR)** holds your tensors. It is big and has high bandwidth,
  but every access takes hundreds of nanoseconds to come back.
- **Registers** and **shared memory (smem)** live inside each SM. They are fast but small:
  a T4 SM has 256 KB of registers and up to 64 KB of shared memory.
- **The L2 cache** sits between the SMs and global memory. Data that one SM reads can be
  served from L2 when another SM asks for it soon after.

Inside an SM, threads run in groups of 32 called **warps**. All 32 threads of a warp run the
same instruction at the same time on different data. This is SIMT: single instruction,
multiple threads.

### 2. The Triton programming model: programs, not threads

In CUDA you write the code for one thread. In Triton you write the code for one
**program**, and a program works on a whole block of data at once.

- You launch a **grid** of programs: `add_kernel[grid](...)`. Here `grid = (cdiv(n, BLOCK_SIZE),)`,
  so there is one program per `BLOCK_SIZE` elements, rounded up.
- Each program asks "which one am I?" with `tl.program_id(axis=0)`. That number, `pid`,
  is the only thing that differs between programs.
- Inside a program, values are **blocks** (Triton calls them tensors): `tl.arange(0, BLOCK_SIZE)`
  is the vector `[0, 1, ..., BLOCK_SIZE-1]`, and `x + y` adds two such vectors lane by lane.
  There are no per-element loops.
- Triton decides how the block maps to threads. With `num_warps=4` a program runs on
  4 x 32 = 128 threads, so a `BLOCK_SIZE = 1024` block gives each thread 8 elements.
  You choose the block size and the warp count; the compiler handles the rest.

![n = 2500 and BLOCK_SIZE = 1024 give 3 programs: pid 0 covers elements 0 to 1023, pid 1 covers 1024 to 2047, and pid 2 covers 2048 to 3071, where only 2048 to 2499 are real; its last 572 lanes are past the end of the tensor and are masked off](figures/grid-mask.svg)

Programs can run in any order and at the same time. Never assume program 0 finishes before
program 1 starts.

### 3. Offsets and pointers

A tensor argument arrives in the kernel as a **pointer** to its first element. Adding an
integer moves it by that many elements (not bytes), as in C. Adding a **vector** of integers
gives a **vector of pointers**, one per lane:

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # e.g. pid 1 -> [1024, 1025, ..., 2047]
x_ptr + offsets                                         # 1024 pointers into x
```

`tl.load(pointers)` reads every address in one operation and returns a block of values.
`tl.store(pointers, values)` writes them.

### 4. Masks: the last block is usually partial

`n` is rarely a multiple of `BLOCK_SIZE`, so the last program has lanes past the end of the
tensor (pid 2 in the picture above). Reading there reads someone else's memory; writing there
corrupts it or crashes.

A **mask** is a block of booleans, one per lane, that says which lanes are real. Loads and
stores skip the lanes where it is `False`. Every kernel in this repo has masks, and a wrong
mask is the most common bug you'll hit. The tests use sizes that are deliberately not
multiples of the block size so that a missing mask fails.

`tl.load(..., mask=m, other=v)` also fills the skipped lanes with `v` (default undefined).
For vector add it doesn't matter, because those lanes are never stored. From exercise 3 on,
`other=` matters a lot.

### 5. Arithmetic intensity and the roofline

Every kernel does two kinds of work: it moves bytes and it does math. Which one limits it?

**Arithmetic intensity (AI)** = FLOPs / bytes moved to and from global memory.

For vector add in fp16, each element reads 2 bytes of `x`, 2 bytes of `y` and writes 2 bytes
of `out`, and does 1 add:

```text
AI = 1 FLOP / 6 bytes ≈ 0.17 FLOP/byte
```

The GPU has two limits: peak compute (FLOP/s) and peak bandwidth (bytes/s). Their ratio is
the **ridge point**:

```text
T4:   65 TFLOP/s  /  320 GB/s   ≈ 203 FLOP/byte
H100: 989 TFLOP/s / 3350 GB/s   ≈ 295 FLOP/byte
```

The **roofline** says: attainable FLOP/s = min(peak compute, AI x bandwidth).

![Log-log T4 roofline: a bandwidth line of slope 320 GB/s meets the 65 TFLOP/s compute roof at the ridge, about 203 FLOP/byte; vector add sits far left at AI 0.17, capped at about 53 GFLOP/s, in the memory-bound region](figures/roofline.svg)

A kernel to the left of the ridge is **memory-bound**: its speed is set by bytes/s, and
doing more math per byte is free. A kernel to the right is **compute-bound**. Vector add is
about 1000x below the ridge, so its only performance metric is GB/s. The `report` line
prints `% of roof`: achieved time compared with the time the roofline allows.

### 6. Why big blocks and many bytes in flight

Global memory has high bandwidth but high latency (roughly 400 to 800 ns). To keep the
memory system busy you need many requests in flight at once. This is **Little's law**:

```text
bytes in flight = bandwidth x latency ≈ 320 GB/s x 500 ns ≈ 160 KB   (T4)
```

That is about 4 KB in flight per SM, all the time. A GPU gets there by running many
programs per SM and by having each program issue large loads. If blocks are tiny, each
program does a little work and then pays its fixed overhead (scheduling, index math, the
tail of the last memory request). If blocks are huge, there are few programs and some SMs
may sit idle at the end. The benchmark sweeps `BLOCK_SIZE` so you can see where the sweet
spot is on your GPU (the first stretch goal adds `num_warps`).

### 7. Launch overhead

Every kernel launch costs a few microseconds of CPU and driver work, whatever its size. At
n = 65,536 fp16 elements, vector add moves 393 KB, which takes about 1.2 µs at 320 GB/s, so
the launch costs more than the work. Small kernels can't reach the roof. This is one reason
fusion (exercises 2, 3, 5) matters: fewer launches as well as fewer bytes.

### 8. How the tests run

On your laptop, `uv run poe test 1` runs the kernel on Triton's **CPU interpreter** in fp32.
It executes each program in turn with NumPy, so it checks correctness but says nothing about
speed. Benchmarks need a real GPU, which is why they run in Colab.

## What to look for

- **n = 65k:** everyone is slow, including PyTorch. The kernel takes a few microseconds and
  most of that is launch overhead (section 7).
- **n = 16M and above:** a good kernel reaches 80-90% of peak bandwidth. That number is your
  GPU's **practical memory roof**. Write it down; later exercises are graded against it.
- **Block sizes:** tiny blocks (128) mean too many programs and too few bytes in flight per
  program. Note which `BLOCK_SIZE` wins, and by how much.
- **Triton vs torch:** for large n they should be within a few percent of each other. Both
  are hitting the same wall: memory bandwidth.

## Check yourself

Answer these in your notes before moving on.

1. For n = 10,000 and `BLOCK_SIZE = 1024`, how many programs run? How many lanes of the
   last program are masked off?
2. What is the arithmetic intensity of vector add in fp32? Does the kernel get faster or
   slower in fp32, and by how much?
3. Why does `alpha * x + y` take the same time as `x + y`?
4. Your kernel runs at 270 GB/s on a T4. What fraction of the roof is that? Is it worth
   optimizing further?
5. Why can't you write `if offsets < n_elements:` inside the kernel instead of using a mask?

## Stretch

- Pass `num_warps=` to the launch (`add_kernel[grid](..., num_warps=8)`) and see if it moves.
- Write `add_scaled(x, y, alpha)` computing `alpha * x + y`. Does the time change? Why not?
- Compute the time a perfect kernel would take at n = 2^27 on your GPU from the spec-sheet
  bandwidth. How far is your best result from it?

## Key takeaways

- A Triton kernel is: program id → offsets → mask → load → compute → store.
- Pointers plus a vector of offsets give a vector of pointers; loads and stores are whole blocks.
- Always mask. The last block is usually partial.
- Arithmetic intensity tells you which roof you're under. Vector add is pure bandwidth.
- Your practical bandwidth roof is ~80-90% of the spec sheet. Small kernels are dominated by
  launch overhead.
