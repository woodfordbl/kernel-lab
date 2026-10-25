# Hints: open one at a time

<details><summary>Hint 1: Part A padding</summary>

Load with `other=-float("inf")`. Then `exp(-inf - m) = 0`, so padding adds nothing to the
sum, and -inf can never be the max of a row that has at least one real value.
</details>

<details><summary>Hint 2: Part B tiles</summary>

The K tile, transposed: `k_ptr + base + offs_n[None, :] * stride_n + offs_d[:, None]`, which
has shape [D, BLOCK_N]. V is the natural layout: `offs_n[:, None] * stride_n + offs_d[None, :]`.
Mask both with `offs_n < N` broadcast the right way, `other=0.0`.
</details>

<details><summary>Hint 3: Part B masking and the end</summary>

```
valid = offs_n[None, :] < N
if CAUSAL:
    valid = valid & (offs_n[None, :] <= offs_m[:, None])
s = tl.where(valid, s, float("-inf"))
```
Rowmax is `tl.max(s, axis=1)`, rowsum is `tl.sum(p, axis=1)`. After the loop:
`acc = acc / l_i[:, None]`, then store with the same pointer pattern as the Q load.
</details>
