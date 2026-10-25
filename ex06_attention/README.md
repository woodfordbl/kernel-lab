# Exercise 6: softmax, then FlashAttention

**Goal:** understand the online softmax well enough to write it, then build a causal
FlashAttention forward pass and beat a naive implementation by several times.

Attention is the one part of a transformer layer that isn't a plain GEMM or an elementwise
op. Done naively, it writes an N x N matrix per head to memory, and at long sequence lengths
that matrix costs more than everything else. FlashAttention removes it entirely. It's
probably the most important kernel of the last few years, and after exercises 1 to 5 you have
every piece you need to write it.

**Builds on:** exercise 3 (row reductions, padding with the reduction's identity) for Part A,
and exercise 4 (tiles, `tl.dot`, the K loop) for Part B.

## Do

1. Read Concepts 1 to 3. Part A, `softmax.py`: a one-pass row softmax.
   `uv run poe test 6 -k softmax`
2. Read Concepts 4 to 8. Work the example in Concept 5 by hand before you code anything.
3. Part B, `kernel.py`: FlashAttention forward. Start with `causal=False`, get those tests
   green, then make causal pass. `uv run poe test 6`
4. `python -m ex06_attention.bench` in Colab.

## Concepts

### 1. Attention in one line

For one head, with N tokens and head dimension D:

```text
S = Q @ K^T * scale        [N, N]   scores: how much query i attends to key j
P = softmax(S, along rows) [N, N]   each row sums to 1
O = P @ V                  [N, D]   each output row is a weighted average of V's rows

Q, K, V, O: [B, H, N, D]           scale = 1 / sqrt(D)
```

The `1/sqrt(D)` keeps scores from growing with D, which would make the softmax too peaked.

**Causal** attention (decoder models) lets query i see only keys j ≤ i. Scores with j > i are
set to `-inf` before the softmax, so they get probability 0.

### 2. Softmax and numerical stability

```text
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

Computed directly, `exp` overflows: `exp(89)` is already past fp32's max, and fp16 overflows
at `exp(11.1)`. The fix is to subtract the row max first:

```text
softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m),     m = max_j x_j
```

This gives the same answer (the `exp(-m)` factor cancels top and bottom), and now every
exponent is ≤ 0, so every `exp` is in (0, 1].

A row softmax kernel is then three steps on a row held in registers: max (a reduction),
exp and sum (a reduction), divide. It is exercise 3's structure with `tl.max` added.

### 3. Part A: padding for a max

As in exercise 3, `BLOCK_N` is a power of two, so the row has padding lanes. They must:

- never win the max, and
- contribute 0 to the sum of exps.

What `other=` value does both? (For exercise 3's sum it was 0. 0 doesn't work here: a row of
all-negative scores would get max 0.)

### 4. Why naive attention is slow

`S` is N x N per head. At N = 4096, H = 16, fp16, that's 4096² x 16 x 2 B = **537 MB**, and
naive attention writes it, reads it for the softmax, writes `P`, and reads `P` again for the
`@ V`. Roughly 2 GB of traffic, against 4 x 8 MB for Q, K, V and O themselves.

The FLOPs (two GEMMs, `4 N² D` per head) are fine. The traffic is the problem, and it grows as
N² while the useful inputs and outputs grow as N.

![HBM traffic at N = 4096, H = 16, D = 64, fp16: naive attention writes S, reads S, writes P and reads P, 537 MB each, about 2.15 GB in all; FlashAttention moves only the 34 MB of Q, K, V and O](figures/traffic.svg)

FlashAttention's idea: compute `S` and `P` one tile at a time, in registers, and never write
them. To do that, you need a softmax that doesn't have to see the whole row first.

### 5. The online softmax

The problem: to normalize a row you need its max and its sum, which depend on every score in
the row. But the kernel sees the row in chunks of `BLOCK_N` keys.

The trick: keep **running** values and correct them when they turn out to be wrong. For each
query row keep:

```text
m    the largest score seen so far
l    sum of exp(score - m) over the scores seen so far
acc  sum of exp(score - m) * v over the keys seen so far   (the unnormalized output, [D])
```

When a new chunk arrives with a larger max `m_new`, everything accumulated so far was computed
relative to the old `m`. Multiplying by `alpha = exp(m - m_new)` converts it exactly:

```text
exp(s - m) * exp(m - m_new) = exp(s - m_new)
```

So the update for each chunk is: new max, rescale the old `l` and `acc` by `alpha`, add the
chunk's contributions. At the end, `O = acc / l`. This is exact, not an approximation.

**Work it by hand.** One query, scores `[1, 3, 2, 5]`, values `v = [10, 20, 30, 40]` (scalars,
D = 1), split into chunks `[1, 3]` and `[2, 5]`.

```text
chunk 1: scores [1, 3]
  m   = 3
  p   = [e^(1-3), e^(3-3)]           = [0.1353, 1.0]
  l   = 0.1353 + 1.0                 = 1.1353
  acc = 0.1353*10 + 1.0*20           = 21.353

chunk 2: scores [2, 5]
  m_new = max(3, 5)                  = 5
  alpha = e^(3-5)                    = 0.1353
  p     = [e^(2-5), e^(5-5)]         = [0.0498, 1.0]
  l     = 1.1353*0.1353 + 0.0498 + 1.0          = 1.2034
  acc   = 21.353*0.1353 + 0.0498*30 + 1.0*40    = 44.383
  m     = 5

O = acc / l = 44.383 / 1.2034 = 36.88
```

Check against the direct formula: `exp([1,3,2,5] - 5) = [0.0183, 0.1353, 0.0498, 1.0]`, sum
1.2034, weighted sum 44.383, so O = 36.88. Same answer.

![The worked example as bars. After chunk 1 (m = 3) the weights are 0.1353 and 1.0. Chunk 2 raises the max to 5, so the old weights shrink by alpha = 0.1353 to 0.0183 and 0.1353, and the new weights are 0.0498 and 1.0. Then l = 1.2034, acc = 44.383 and O = 36.88](figures/online-softmax.svg)

In the kernel, everything is a block: `m` and `l` have one entry per query row (`[BLOCK_M]`),
`acc` is `[BLOCK_M, D]`, and the row max and row sum are `tl.max` and `tl.sum` along `axis=1`
of the `[BLOCK_M, BLOCK_N]` score block.

### 6. The FlashAttention kernel structure

Grid: `(cdiv(N, BLOCK_M), B * H)`. Each program owns `BLOCK_M` query rows of one (batch, head).

```text
load Q tile [BLOCK_M, D]                     stays in registers for the whole loop
m = -inf, l = 0, acc = 0

for each block of BLOCK_N keys:
    load K tile as [D, BLOCK_N]              (transposed by the pointer layout, ready for dot)
    load V tile as [BLOCK_N, D]
    s = Q @ K_tile * scale                   [BLOCK_M, BLOCK_N]   tensor cores
    mask invalid scores to -inf
    online softmax update of m, l, acc       acc gets P_tile @ V_tile, tensor cores again

O = acc / l, store
```

![One program's key loop: the Q tile for query block 2 stays resident; at each step one K tile and one V tile are loaded, a BLOCK_M x BLOCK_N block of s and p lives only in registers, and p @ V is folded into the fp32 accumulator with running m and l. The N x N score matrix is never stored](figures/flash-loop.svg)

It's exercise 4's K loop with a softmax wedged between two `tl.dot`s. Notice:

- **The K tile is loaded transposed** by building its pointers with D on the first axis and
  keys on the second. No transpose op is needed. (Exercise 2: the address formula doesn't care
  which index you put on which axis.)
- **`P` is cast to V's dtype** (fp16) before the second dot, because `tl.dot` wants fp16
  inputs. The accumulator stays fp32.
- **Scores never leave registers.** `s` and `p` exist only inside the loop. HBM traffic is Q,
  K, V and O, the same as an elementwise op on each (plus re-reads of K and V by different
  query blocks, which mostly hit L2).

### 7. Masking with `-inf`

Two kinds of score are invalid and must become `-inf` before the softmax, so `exp` sends them
to 0:

- keys past N (the last key block is partial), and
- with `CAUSAL`, keys after the query: key index > query index.

`tl.where(valid, s, float("-inf"))` does it. The 2-D `valid` block comes from broadcasting a
query-index column against a key-index row, as in exercise 2.

One subtlety: if a row's running max is still `-inf` when a chunk with all scores `-inf`
arrives, the update computes `exp(-inf - (-inf)) = NaN`. Here that can't happen: every
query can see key 0, so the first chunk always gives each row a finite max. Variants where a
row can start with no visible keys (sliding windows, padding masks) need a guard, and
production kernels have one.

### 8. Causal skipping

For query block i, every key block after block i is entirely masked: all its keys come after
all of block i's queries. There's no point loading them. The loop bound `hi` stops there,
which halves the total work. Only the diagonal block (where keys and queries overlap) needs
per-element masking.

![A 4 x 4 map of query blocks against key blocks: blocks below the diagonal are full and need no mask, diagonal blocks need the per-element mask, and blocks above the diagonal are skipped because the loop stops at hi. Zoomed in, a diagonal block keeps scores where key index <= query index and sets the rest to -inf](figures/causal-blocks.svg)

### 9. The decode case

At decode there is 1 new query per sequence. With this kernel, each program has `BLOCK_M = 64`
query slots and only 1 is real, and the grid is just `B * H` programs, often fewer than the
number of SMs. Each program then streams the whole KV cache alone. **Flash-Decoding** fixes
this by splitting the keys across programs (split-K over N) and combining the partial
`(m, l, acc)` triples with the same rescaling rule in a second small kernel.

## What to look for

- **Naive vs yours:** the gap grows with N, since naive traffic grows as N².
- **SDPA** uses a production kernel (FlashAttention or memory-efficient attention, depending on
  the GPU). Getting within 1.5x of it is a good result for a first version.
- **Block sizes:** the bench tries 64x64, 128x64 and 64x32. Larger `BLOCK_M` reuses each K/V tile
  across more queries; larger `BLOCK_N` means fewer loop steps but more registers for `s`.
- **The "% of roof" column** uses the ideal IO and the causal-half FLOPs. Attention at N = 4096,
  D = 64 is compute-bound, so look at TFLOPS.

## Check yourself

1. Why does subtracting the row max not change the softmax?
2. In the online update, why are `l` and `acc` multiplied by `alpha` but `p` is not?
3. What `other=` value do you load padding with in Part A, and why does it satisfy both
   requirements?
4. How many bytes does naive attention's score matrix take at N = 8192, H = 32 in fp16?
5. For causal attention with N = 1024 and `BLOCK_M = BLOCK_N = 64`, how many K/V blocks does
   query block 5 load? How many need per-element masking?
6. What does `m` start at, and why does the first update work even though `exp(m - m_new)` is
   `exp(-inf)`?

## Stretch

- Replace `tl.exp` with `tl.exp2` and fold `log2(e)` into `scale`. Production kernels do this
  because exp2 maps to one hardware instruction.
- Split the loop in two: off-diagonal blocks with no masking, then the diagonal block with
  the causal mask. Measure the gain.
- Decode case: 1 query, N = 4096 keys. Why would this kernel use the GPU badly, and how does
  Flash-Decoding fix it? (Concept 9; `AGENTS.md` lists it as a future exercise.)
- Also store the log-sum-exp `m + log(l)` per row. The backward pass needs it.

## Key takeaways

- Softmax needs the row max for stability; padding for a max uses `-inf`.
- Naive attention's cost is the N x N score matrix's traffic, not its FLOPs.
- The online softmax keeps running `(m, l, acc)` and rescales by `exp(m_old - m_new)` when the
  max grows. It is exact.
- FlashAttention = exercise 4's loop: Q tile resident, stream K and V tiles, two `tl.dot`s per
  step, softmax in registers in between. HBM traffic is O(N), not O(N²).
- Causal attention skips blocks above the diagonal and masks only the diagonal block.
