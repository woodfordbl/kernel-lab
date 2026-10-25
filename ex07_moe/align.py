"""Exercise 7 helper: sort (token, expert) pairs by expert and pad each expert's group.

This is given to you; read it before writing the kernel. It is the "histogram, prefix sum,
permute" step from the lesson, written with PyTorch ops. Stretch: rewrite it in Triton.

Input:  topk_ids [T, top_k], the experts each token picked.
Output:
  sorted_ids   [P]  indices into the flattened [T * top_k] pairs, grouped by expert, each
                    group padded to a multiple of block_m with the sentinel value T * top_k
  expert_ids   [P // block_m]  which expert owns each block of block_m rows
  P is the padded total. Every block of block_m consecutive rows belongs to exactly one
  expert, which is what lets one GEMM tile use one expert's weights.

Example: T=3, top_k=2, block_m=4, topk_ids = [[0, 2], [2, 1], [0, 2]]
  flattened pairs: 0->e0, 1->e2, 2->e2, 3->e1, 4->e0, 5->e2
  sorted_ids = [0, 4, 6, 6,   3, 6, 6, 6,   1, 2, 5, 6]     (6 = sentinel)
  expert_ids = [0,            1,            2]
"""
import torch


def moe_align(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    flat = topk_ids.flatten().long()
    n_pairs = flat.numel()
    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block_m - 1) // block_m * block_m
    starts = torch.cumsum(padded, 0) - padded            # where each expert's group begins
    first = torch.cumsum(counts, 0) - counts             # rank offset of each expert in sorted order

    order = torch.argsort(flat, stable=True)             # pair indices sorted by expert
    e_sorted = flat[order]
    rank_in_expert = torch.arange(n_pairs, device=flat.device) - first[e_sorted]
    total = int(padded.sum())
    sorted_ids = torch.full((total,), n_pairs, dtype=torch.int32, device=flat.device)
    sorted_ids[starts[e_sorted] + rank_in_expert] = order.to(torch.int32)

    expert_ids = torch.repeat_interleave(torch.arange(num_experts, device=flat.device),
                                         padded // block_m).to(torch.int32)
    return sorted_ids, expert_ids
