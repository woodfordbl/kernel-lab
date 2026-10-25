# Exercise 5: GEMM with a fused SwiGLU epilogue

**Goal:** fuse an elementwise op into a GEMM's epilogue, and see when fusion matters and
when it doesn't.

Exercise 2 fused the SwiGLU activation into one kernel, but its inputs still came from a
GEMM that wrote them to memory. Here the activation moves **inside** the GEMM: the gate and
up projections and SiLU * up happen in one kernel, and the `[M, 2F]` intermediate never
exists in memory at all.

**Builds on:** exercise 4 (the tiled GEMM, the K loop, grouped ordering) and exercise 2 (the
SwiGLU math, the packed `[K, 2F]` layout, fusion accounting).

## Do

1. Read the Concepts below.
2. Fill in TODOs 1 to 4. The tile mapping and x pointers are given (you wrote them in ex04).
3. `uv run poe test 5`, then `python -m ex05_matmul_swiglu.bench` in Colab.

## Concepts

### 1. The operation

```text
out[M, F] = SiLU(x @ W_gate) * (x @ W_up)        x: [M, K]   W_gate, W_up: [K, F]
```

This is the first half of the MLP from exercise 2's diagram. With K = 4096 and F = 14336
(Llama-3-8B), it's the single most expensive operation in the layer.

### 2. The unfused baseline

How PyTorch (and most simple implementations) run it:

```text
gu  = x @ W_cat                     one GEMM, W_cat = [W_gate | W_up] of shape [K, 2F]
                                    writes gu: [M, 2F]
out = silu(gu[:, :F]) * gu[:, F:]   exercise 2: reads gu back, writes out
```

The extra traffic, beyond the minimum, is writing `gu` and reading it back:
`2 * M * 2F * 2 bytes` and more in eager, which re-reads parts of it. At M = 4096 that's
roughly 0.5 to 0.7 GB, about 2 ms at T4 bandwidth.

![Unfused: a GEMM kernel reads x and W_cat from HBM and writes gu [M, 2F] back to HBM, then a second kernel reads gu again and writes out. Fused: one kernel reads x and the weights, applies SiLU(acc_gate) * acc_up in its epilogue, and writes only out; gu never exists in memory](figures/unfused-vs-fused.svg)

### 3. Epilogue fusion

Look at the end of exercise 4's kernel: when the K loop finishes, the program holds a
finished `BLOCK_M x BLOCK_N` tile of results in fp32 registers. The **epilogue** is
everything between that point and the store. Any elementwise function of the accumulator is
almost free there:

- bias add, scaling, activation (ReLU, GELU, SiLU)
- residual add (one extra tile load)
- casting to fp8 or int8 for the next layer, with a scale
- multiplying by a router weight (exercise 7 does this)

It saves a full write and re-read of the output, plus a kernel launch. Production GEMM
libraries (cuBLASLt, CUTLASS) expose epilogues for exactly this reason.

SwiGLU is a little harder than a plain activation because it combines **two** GEMM outputs
elementwise: `out[i, j]` needs `gate[i, j]` and `up[i, j]`. So one program must produce the
matching tile of both.

### 4. Two accumulators, one x tile

Each program owns output tile `(pid_m, pid_n)` and keeps two accumulators for it:

![One program's K loop: at each step it loads one x tile once, plus the W_gate and W_up tiles at the same offsets, and adds both products to two fp32 accumulators, acc_gate and acc_up. The gate and up results exist only in registers. After the loop the epilogue computes out_tile = SiLU(acc_gate) * acc_up, casts it and stores it; out is the only thing written](figures/two-accumulators.svg)

`W_gate` and `W_up` have the same shape and strides, so their tile pointers use the same
offsets with different base pointers. And they share one mask.

**Operand reuse:** each x tile is loaded once and feeds two `tl.dot` calls. Two separate
GEMMs would load every x tile twice, so this halves x traffic. (One wide GEMM over `W_cat`
gets the same reuse, which is why real models store the weights packed.)

### 5. Register pressure

Registers are the scarcest resource on an SM. A T4 SM has 65,536 32-bit registers shared by
all the threads running on it, and each thread can use at most 255.

Two fp32 accumulators double what the K loop must keep alive:

![Accumulator registers per thread with 8 warps (256 threads), against the 255-per-thread cap. Exercise 4's 128 x 128 tile with one accumulator: 128 * 128 / 256 = 64. The same tile with two accumulators: 2 * 128 * 128 = 32,768 fp32 values, 128 per thread, before A/B fragments, pointers, masks and loop counters. A 128 x 64 tile with two accumulators: 2 * 128 * 64 = 16,384 values, back to 64 per thread](figures/registers.svg)

Two things go wrong when register use climbs:

- **Lower occupancy.** Fewer programs fit on an SM at once, so there is less work to switch
  to while loads are pending, and latency hiding suffers.
- **Spilling.** If a thread needs more than 255 registers, the compiler moves values to
  "local memory", which is private per thread but physically lives in device memory behind
  the caches. Every spill is a slow load or store inside the hot loop, and performance falls
  off a cliff.

That's why this exercise's configs are a step smaller than exercise 4's. When a kernel is
inexplicably slow, register spills are one of the first things to check (Triton can print
the compiled kernel's register and spill counts; see the `n_regs` and `n_spills` attributes
of a compiled kernel).

### 6. When fusion pays: count the bytes it removes

Fusion is not automatically a big win. It saves the bytes of the intermediate and nothing
else. Compare that with the total.

```text
M = 4096 (prefill)
  GEMM FLOPs: 2 * 4096 * 4096 * 28672 ≈ 962 GFLOP  -> ~15 ms at T4 peak, more in practice
  intermediate traffic removed: ~0.5-0.7 GB          -> ~2 ms
  => a clear, but not dramatic, win (on the order of 10-15%)

M = 16 (decode)
  weights: 2 * 4096 * 14336 * 2 B ≈ 235 MB             -> ~0.8 ms, and this dominates
  intermediate: 16 * 28672 * 2 B ≈ 0.9 MB              -> negligible
  => fusion saves almost nothing; any difference comes from tile choice and launch count
```

![Each bar is the unfused cost on a T4, split into what fusion removes and what remains. At M = 4096, about 15 ms of GEMM compute remains and fusion removes about 2 ms of intermediate traffic, 10 to 15 percent. At M = 16, reading 235 MB of weights (about 0.8 ms) dominates and the 0.9 MB intermediate is under 1 percent](figures/fusion-payoff.svg)

The rule: **fusion pays off in proportion to the bytes it removes, relative to the bytes
(or FLOPs) that remain.** In exercise 2 the intermediate was 40% of all traffic, so fusion
gave 1.6x. Here, at decode, it's well under 1%.

### 7. The SiLU math in the epilogue

`SiLU(g) = g * sigmoid(g)`. The accumulators are already fp32, so compute it in fp32 and cast
once at the store, exactly as in exercise 2. The store uses `F` in place of exercise 4's `N`
and the output strides.

## What to look for

- **M = 4096:** the GEMM is compute-bound, and the unfused version's extra traffic (writing and
  re-reading `[4096, 28672]`) is a real fraction of the time. Expect a clear win.
- **M = 16:** the weights dominate everything. The intermediate is tiny, so fusion saves
  little. Your kernel may even lose to cuBLAS here if its tiles waste rows (how many rows of a
  128-row tile are real at M = 16?).
- **Compare TFLOPS with exercise 4's prefill number.** Same FLOPs, so the fused kernel should be
  close to it or better. If it's much worse, suspect register pressure.

## Check yourself

1. Why can't the SiLU be applied inside the K loop, after each `tl.dot`?
2. How many bytes of HBM traffic does fusion remove at M = 4096, F = 14336 in fp16?
3. A 64 x 128 tile with two accumulators and 4 warps: how many accumulator registers per thread?
4. Name three other epilogues worth fusing into a GEMM in a transformer layer.
5. At M = 16, which config would you expect autotune to pick, and why?

## Stretch

- Use the packed `[K, 2F]` layout directly: W_up's tile is W_gate's tile offset by F columns.
- Interleave gate and up columns (gate0, up0, gate1, up1, ...) so one `tl.dot` of width
  2*BLOCK_N computes both. What would you need to split the accumulator afterward?
- Print `n_regs` and `n_spills` for each config. Add a 128 x 128 config with 4 warps and watch
  what happens.

## Key takeaways

- The epilogue is where elementwise work on a GEMM's output is nearly free.
- Combining two GEMM outputs needs two accumulators per program, sharing each x tile.
- Accumulators live in registers. Too many registers → lower occupancy, then spills to slow
  local memory. Shrink tiles when you add accumulators.
- Fusion's payoff = bytes removed / bytes remaining. Large at prefill, negligible at decode.
