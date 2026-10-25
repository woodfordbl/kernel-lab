# Exercise 6: softmax, then FlashAttention

**Goal:** understand the online softmax well enough to write it, then build a causal
FlashAttention forward pass and beat a naive implementation by several times.

## Do

1. Part A, `softmax.py`: a one-pass row softmax. `uv run poe test 6 -k softmax`
2. Part B, `kernel.py`: FlashAttention forward. Start with `causal=False`, get those tests
   green, then make causal pass. `uv run poe test 6`
3. `python -m ex06_attention.bench` in Colab

## Concepts

- **Why naive attention is slow.** `Q @ K^T` produces an N x N matrix per head. At N = 4096
  and 16 heads that's 537 MB in fp16, written, re-read by softmax, written again, and re-read
  by the `@ V`. The FLOPs are fine; the traffic is the problem.
- **Online softmax.** You can't softmax a row until you've seen its max, but you can keep a
  running max and fix up earlier work when it changes. Rescaling by `exp(m_old - m_new)` is
  exact, not an approximation. Work it through by hand for a row split into two chunks before
  you code it.
- **Causal skipping.** For query block i, key blocks after block i are entirely masked. The
  loop bound `hi` skips them, which halves the work. Only the diagonal block needs per-element
  masking.
- **Scores never leave registers.** `s` and `p` exist only inside the loop. The kernel's
  HBM traffic is Q, K, V and O, the same as one elementwise op on each.

## What to look for

- Naive vs yours: the gap grows with N, since naive traffic grows as N².
- SDPA uses a production kernel (FlashAttention or memory-efficient attention, depending on
  the GPU). Getting within 1.5x of it is a good result for a first version.

## Stretch

- Replace `tl.exp` with `tl.exp2` and fold `log2(e)` into `scale`. Production kernels do this
  because exp2 maps to one hardware instruction.
- Split the loop in two: off-diagonal blocks with no masking, then the diagonal block with
  the causal mask. Measure the gain.
- Decode case: 1 query, N = 4096 keys. Why would this kernel use the GPU badly, and how does
  Flash-Decoding fix it? (Lesson Part 3.)
