# Hints: open one at a time

<details><summary>Hint 1: pointer tiles</summary>

The same broadcasting trick as exercise 2:
`a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak`.
B is `[K, N]`, so its tile uses `offs_k[:, None]` and `offs_n[None, :]`.
</details>

<details><summary>Hint 2: the K loop</summary>

The K-tail mask uses how much K is left: `offs_k[None, :] < K - k0` for A and
`offs_k[:, None] < K - k0` for B, combined with the M or N mask. Load with `other=0.0` so
padding contributes nothing to the dot. Advance with `a_ptrs += BLOCK_K * stride_ak`.
</details>

<details><summary>Hint 3: grouped ordering, nearly the answer</summary>

```
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```
</details>
