# Hints: open one at a time

<details><summary>Hint 1: Part A</summary>

Copy your exercise 1 kernel. Only the line between load and store changes, plus `.to()`
casts on the way in and on the way out.
</details>

<details><summary>Hint 2: Part B pointers</summary>

`rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)` and the same for `cols` with `pid_f` and
BLOCK_F. The gate tile is at `gu_ptr + rows[:, None] * stride_gu_row + cols[None, :]`.
Where is the up tile, relative to that?
</details>

<details><summary>Hint 3: Part B, nearly the answer</summary>

`up_ptrs = gate_ptrs + F`. The output tile uses `stride_out_row` instead of `stride_gu_row`.
Use the same 2-D mask for both loads and the store.
</details>
