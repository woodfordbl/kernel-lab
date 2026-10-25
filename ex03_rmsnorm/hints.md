# Hints: open one at a time

<details><summary>Hint 1: addressing a row</summary>

Row `row` starts at `x_ptr + row * stride_row`. Add `cols` to get the vector of addresses.
Residual, out and h use the same stride (they're all contiguous [T, D]).
</details>

<details><summary>Hint 2: the reduction</summary>

`ms = tl.sum(h * h, axis=0) / D`. This only works if the padding lanes of `h` are 0, so load
both inputs with `other=0.0`.
</details>

<details><summary>Hint 3: nearly the answer</summary>

```
r = tl.rsqrt(ms + eps)
w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
tl.store(out_ptr + row * stride_row + cols, (h * r * w).to(...), mask=mask)
```
</details>
