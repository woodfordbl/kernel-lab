# Hints: open one at a time

<details><summary>Hint 1: the gathered rows</summary>

```
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
pair = tl.load(sorted_ids_ptr + offs_m)
row_mask = pair < num_pairs
token = pair // top_k
```
No mask is needed on that first load: sorted_ids is always a whole number of blocks long.
</details>

<details><summary>Hint 2: pointers</summary>

`a_ptrs = x_ptr + token[:, None] * stride_xm + offs_k[None, :] * stride_xk`.
For B: `w_ptr + e * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn`,
with `e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)`. The K loop is ex04's, with
`row_mask[:, None]` in place of the M mask.
</details>

<details><summary>Hint 3: the epilogue</summary>

```
if MUL_ROUTED_WEIGHT:
    rw = tl.load(topk_w_ptr + pair, mask=row_mask, other=0.0)
    acc = acc * rw[:, None]
out_ptrs = out_ptr + pair[:, None] * stride_om + offs_n[None, :] * stride_on
```
Store with `row_mask[:, None] & (offs_n[None, :] < N)`.
</details>
