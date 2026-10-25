# Hints: open one at a time

<details><summary>Hint 1: the shape of every Triton elementwise kernel</summary>

Every program computes a vector of indices, loads a vector, does math on the whole vector
at once, and stores a vector. There are no per-element loops. `offsets` is a vector of
BLOCK_SIZE integers, and `x_ptr + offsets` is a vector of pointers.
</details>

<details><summary>Hint 2: the mask</summary>

`mask = offsets < n_elements` is a vector of booleans. Pass it as `mask=mask` to both
`tl.load` and `tl.store`. Out-of-range lanes are skipped instead of touching memory you
don't own.
</details>

<details><summary>Hint 3: nearly the answer</summary>

```python
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = ...
x = tl.load(x_ptr + offsets, mask=mask)
...
```
</details>
