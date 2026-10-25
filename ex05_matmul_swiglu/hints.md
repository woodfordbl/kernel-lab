# Hints: open one at a time

<details><summary>Hint 1: two of everything</summary>

`wg_ptrs` and `wu_ptrs` are built exactly like ex04's `b_ptrs`, with the same offsets and
strides. Only the base pointer differs. Same for the two accumulators.
</details>

<details><summary>Hint 2: the loop body</summary>

One `x = tl.load(x_ptrs, mask=..., other=0.0)`, then `g = tl.load(wg_ptrs, ...)` and
`u = tl.load(wu_ptrs, ...)` with the same mask, then `acc_g += tl.dot(x, g)` and
`acc_u += tl.dot(x, u)`. Advance all three pointer tiles.
</details>

<details><summary>Hint 3: the epilogue</summary>

`h = acc_g * tl.sigmoid(acc_g) * acc_u`, then store it like ex04's C tile, with F in place
of N and the `stride_om` / `stride_on` strides.
</details>
