# Exercise 5: GEMM with a fused SwiGLU epilogue

**Goal:** fuse an elementwise op into a GEMM's epilogue, and see when fusion matters and
when it doesn't. This is steps 8 + 9 of the Layer Trace in one kernel.

## Do

1. Fill in TODOs 1-4. The tile mapping and x pointers are given (you wrote them in ex04).
2. `uv run poe test 5`, then `python -m ex05_matmul_swiglu.bench` in Colab

## Concepts

- **Epilogue fusion.** The accumulator is already in registers when the K loop ends.
  Anything elementwise on it (bias, activation, scaling, residual add, a cast to FP8) is
  nearly free there, and it saves writing and re-reading the whole output.
- **Operand reuse.** Each x tile is loaded once and feeds two `tl.dot` calls. That raises
  the intensity of the x traffic by 2x compared with two separate GEMMs.
- **Register pressure.** Two BLOCK_M x BLOCK_N fp32 accumulators per program. If the tile is
  too big, the compiler spills registers to "local memory", which lives in device memory
  behind the caches, and performance falls off a cliff. That's why these configs are smaller than ex04's.

## What to look for

- At M = 4096 the GEMM is compute-bound, so the unfused version's extra traffic (writing and
  re-reading `[4096, 28672]`) is a real fraction of the time. Expect a clear win.
- At M = 16 the weights dominate everything. The intermediate is tiny, so fusion saves
  little. The lesson: fusion pays off in proportion to the bytes it removes.

## Stretch

- Use the packed `[K, 2F]` layout directly: W_up's tile is W_gate's tile offset by F columns.
- Interleave gate and up columns (gate0, up0, gate1, up1, ...) so one `tl.dot` of width
  2*BLOCK_N computes both. What would you need to split the accumulator afterward?
