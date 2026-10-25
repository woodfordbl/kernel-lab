# Exercise 7: fused-MoE grouped GEMM

**Goal:** write the core kernel of every MoE serving engine: one launch that runs all
experts, gathers token rows on the fly, and scatters weighted results back.

Mixture-of-experts models (Mixtral, DeepSeek, Qwen-MoE, many frontier models) replace the
single MLP with many smaller "expert" MLPs and send each token to only a few of them. That
makes the model much larger at the same compute per token, but it turns one big GEMM into
dozens of small, unevenly sized ones. This exercise is about running them efficiently.

**Builds on:** exercise 4 (the tiled GEMM), exercise 5 (epilogue work on the accumulator) and
exercise 2 (pointer tiles built from index vectors).

## Do

1. Read Concepts 1 to 4. Read `align.py` and work its example by hand.
   `uv run poe test 7 -k align` already passes.
2. Read Concepts 5 to 8. Fill in `moe_gemm_kernel` (four TODOs). `uv run poe test 7 -k moe_gemm`
3. `moe_mlp` is wired up around your kernel. `uv run poe test 7` runs it end to end.
4. `python -m ex07_moe.bench` in Colab.

## Concepts

### 1. What an MoE layer does

```text
x [T, D] ──> router: logits = x @ W_router  [T, E]
             probs = softmax(logits); pick top_k experts per token, renormalize their weights

for each token t, for each of its top_k experts e with weight w:
    y_t += w * Expert_e(x_t)          Expert_e = a SwiGLU MLP with its own W13_e, W2_e

W13: [E, D, 2F]   (gate | up, packed as in exercise 5)
W2:  [E, F, D]
```

The bench uses D = 4096, E = 64 experts, top_k = 4, F = 2048. Each token does the work of 4
small MLPs, but the layer holds 64 of them: ~3.2 GB of weights in fp16.

A **pair** is one (token, chosen expert) combination. There are `T * top_k` pairs, numbered
`pair = token * top_k + j`, so `token = pair // top_k`.

### 2. The obvious implementation and why it's slow

```text
for e in range(E):
    rows = tokens routed to e        # a gather
    h = x[rows] @ W13[e]             # small GEMM
    y = silu(h_gate) * h_up @ W2[e]  # more small kernels
    out.index_add_(rows, y * w)      # a scatter
```

With 64 experts that's hundreds of tiny kernel launches, each with launch overhead (exercise
1), most with too few rows to fill the GPU (exercise 3's occupancy problem), plus a gathered
copy of x per expert written to and read from memory.

### 3. Grouped GEMM: one launch for all experts

Instead, lay out every expert's rows one after another, and launch **one** grid that covers
all of them. The picture uses the small example that Concept 4 works through:

![The 12 sorted rows of the align example form three tiles of BLOCK_M = 4. Program pid_m reads expert_ids[pid_m] and uses that expert's weight slice, w + e * stride_we: tile 0 uses W[0], tile 1 uses W[1], tile 2 uses W[2]. An expert with no pairs gets no tile and its weights are never read](figures/grouped-gemm.svg)

Each program looks up which expert its row-tile belongs to (`expert_ids[pid_m]`) and uses that
expert's weight slice, `w + expert * stride_we`. Experts that got no tokens get no tiles and
cost nothing. Everything else is exercise 4's K loop.

For that to work, each tile must contain rows of only one expert. That's what `align.py`
arranges.

### 4. `moe_align`: sort and pad

`moe_align` is the "histogram, prefix sum, permute" step:

1. **Histogram:** count the pairs routed to each expert.
2. **Pad:** round each count up to a multiple of `BLOCK_M`.
3. **Prefix sum:** where each expert's padded group starts.
4. **Permute:** write each pair's index into its expert's group. Fill the padding with a
   sentinel, `T * top_k`, which is not a valid pair.

The example from `align.py`, with `T = 3`, `top_k = 2`, `block_m = 4`:

![moe_align on topk_ids [[0, 2], [2, 1], [0, 2]]: the six pairs have experts 0, 2, 2, 1, 0, 2. Counts per expert are 2, 1, 3, each padded to 4, so groups start at rows 0, 4, 8. sorted_ids = [0, 4, 6, 6, 3, 6, 6, 6, 1, 2, 5, 6], where 6 = T * top_k is the padding sentinel, and expert_ids = [0, 1, 2]](figures/align.svg)

Check: pairs 0 and 4 went to expert 0, pair 3 to expert 1, pairs 1, 2, 5 to expert 2. Every
block of 4 has one expert. Of 12 rows, 6 are padding.

### 5. Gather in the load

Row r of a tile is pair `sorted_ids[r]`, which reads row `pair // top_k` of x. So the A-tile
pointers are still `row_index[:, None] * stride + col_index[None, :]` (exercise 2), but the row
indices come **from memory** instead of `tl.arange`. The hardware happily loads from scattered
rows: each row is still contiguous along K, so the loads stay coalesced within a row.

No permuted copy of x is ever written. The "permute" step of MoE is fused into the GEMM's
loads.

![Tile pid_m = 2 (expert 2) holds pairs 1, 2, 5 and one padding slot. Its rows are loaded from x rows 0, 1, 2 (token = pair // top_k) and stored to out rows 1, 2, 5 (out[pair]). The padding row, sentinel 6, is masked on both the load and the store](figures/gather-scatter.svg)

Padding rows carry the sentinel. Build a row mask from it (`pair < num_pairs`) and use it
wherever you'd use exercise 4's M mask.

### 6. The expert's weights and 64-bit offsets

`w` is `[E, K, N]`. The expert's slice starts at `w + e * stride_we`, then the B tile is built
as in exercise 4.

`e * stride_we` can overflow 32 bits. Pointer offsets in Triton are computed in the type of the
index values, and `expert_ids` is int32. Here `stride_we = 4096 * 4096 ≈ 16.8M`, so expert 63's
offset is about 1.06 billion, just under 2^31 ≈ 2.15 billion. A model with 128 experts or a
wider hidden size goes past it, and int32 math silently wraps to a wrong address. Casting the
expert id to int64 first avoids it. Keep this in mind for any kernel indexing into a tensor
with more than 2^31 elements.

### 7. Scatter in the store, and the router weight

The output has one row per pair: `out[pair]`. Row r of the tile is stored to `out[sorted_ids[r]]`,
a **scatter**: the same pointer trick as the gather, on the store side. Padding rows are masked
off.

In GEMM 2 (the down projection), each row is also multiplied by its router weight
`topk_w[pair]` in the epilogue (exercise 5's idea). Then the final combine is a plain reshape
`[T, top_k, D]` and sum over `top_k`.

How `moe_mlp` uses your kernel:

```text
GEMM 1:  h = moe_gemm(x, W13)        gathers x rows, writes h[pair]   [T*top_k, 2F]
         h = silu(h_gate) * h_up     (stretch: fuse into GEMM 1, exercise 5)
GEMM 2:  y = moe_gemm(h, W2, *w)     h is already one row per pair, so call it with top_k = 1
                                     (token(pair) = pair); multiply by router weight
combine: out = y.view(T, top_k, D).sum(1)
```

### 8. Padding is wasted work: tokens per expert

Every expert's group is padded to `BLOCK_M`, and padded rows cost full tensor-core work. How
much padding there is depends on **tokens per expert**:

![Drawn to scale. At T = 32, 128 pairs over about 56 touched experts gives about 2.3 rows per expert: a BLOCK_M = 16 tile is about 86% padding and a BLOCK_M = 64 tile about 96%. At T = 4096, 16384 pairs over 64 experts gives about 256 rows per expert: 4 full BLOCK_M = 64 tiles with almost no padding](figures/padding.svg)

At decode the kernel is memory-bound on expert weights: each touched expert's ~50 MB must be
read once regardless (~2.8 GB total, about 9 ms on a T4). Padding rows add no memory traffic
(masked loads don't read), but they add tensor-core work and, more importantly, registers per
program. Fewer programs fit on each SM, so fewer weight loads are in flight (exercise 1's
Little's law), and bandwidth drops. That's why small `BLOCK_M` still wins at decode. At prefill the kernel behaves like a dense GEMM, and big tiles win for exercise 4's
reasons. Production engines (vLLM, SGLang) switch configs on exactly this ratio.

## What to look for

- **T = 32:** memory-bound on expert weights. Compare your GB/s with exercise 1's roof. Small
  `BLOCK_M` should win.
- **T = 4096:** about 256 rows per expert, so this behaves like a dense GEMM. Large `BLOCK_M`
  wins. Compare TFLOPS with exercise 4.
- **The torch loop** pays one launch, one gather and one scatter per expert. With 64 experts
  that's hundreds of small kernels. The gap should be largest at T = 32.

## Check yourself

1. Run `moe_align` by hand for `topk_ids = [[1, 0], [1, 2]]`, `block_m = 2`. What are
   `sorted_ids` and `expert_ids`?
2. Why must every tile belong to exactly one expert?
3. What value marks padding rows, and why is it safe to use as a mask test?
4. At T = 32, roughly how many bytes of expert weights does one MoE layer read? How long is
   that at your measured bandwidth?
5. Why does GEMM 2 pass `top_k = 1`?
6. Where would int32 offset overflow bite first: `e * stride_we` or `token * stride_xm`? Work
   out both for the bench's sizes.

## Stretch

- Fuse SiLU * up into GEMM 1 with two accumulators, as in exercise 5.
- Fuse the combine: in GEMM 2, `tl.atomic_add` each weighted row straight into `out[token]`.
  What does that do to run-to-run determinism? (Floating-point addition is not associative,
  and atomics land in arbitrary order.)
- Rewrite `moe_align` as Triton kernels: histogram with atomics, then a prefix sum.

## Key takeaways

- MoE turns one big GEMM into many small, uneven ones. A grouped GEMM runs them all in one
  launch.
- `moe_align` sorts pairs by expert and pads each group to `BLOCK_M`, so each tile has one
  expert.
- Gather and scatter are just pointer tiles whose row indices come from memory.
- Cast large offsets to int64.
- Padding waste and the best config depend on tokens per expert: small tiles at decode,
  large at prefill.
