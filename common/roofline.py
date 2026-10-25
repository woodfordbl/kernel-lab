"""Roofline numbers and a one-line performance report.

Peaks are vendor spec-sheet numbers for dense FP16/BF16 tensor-core math and HBM/GDDR
bandwidth. Real kernels top out around 80-90% of bandwidth and 70-85% of compute,
so treat ~85% of the roof as "excellent", not 100%.
"""
from dataclasses import dataclass

import torch


@dataclass
class GPUSpec:
    name: str
    peak_flops: float  # FLOP/s, dense fp16 tensor core
    bandwidth: float   # bytes/s, DRAM

    @property
    def ridge(self) -> float:
        return self.peak_flops / self.bandwidth


# (substring of torch.cuda.get_device_name(), peak FLOP/s, bytes/s)
_SPECS = [
    ("H100", 989e12, 3.35e12),
    ("A100", 312e12, 2.0e12),
    ("L4", 121e12, 300e9),
    ("A10", 125e12, 600e9),
    ("T4", 65e12, 320e9),
    ("4090", 165e12, 1.0e12),
    ("3090", 71e12, 936e9),
]


def gpu_spec() -> GPUSpec:
    if not torch.cuda.is_available():
        return GPUSpec("CPU (interpreter)", float("nan"), float("nan"))
    name = torch.cuda.get_device_name()
    for key, f, b in _SPECS:
        if key in name:
            return GPUSpec(name, f, b)
    print(f"[roofline] unknown GPU '{name}', add it to common/roofline.py; using T4 numbers")
    return GPUSpec(name, 65e12, 320e9)


def report(label: str, ms: float, bytes_moved: float, flops: float = 0.0, baseline_ms: float | None = None):
    """Print time, achieved bandwidth/compute and percent of the roofline bound."""
    spec = gpu_spec()
    s = ms / 1e3
    gbps = bytes_moved / s / 1e9
    tflops = flops / s / 1e12
    intensity = flops / bytes_moved if bytes_moved else float("inf")
    roof_s = max(bytes_moved / spec.bandwidth, flops / spec.peak_flops)
    bound = "compute" if flops / spec.peak_flops > bytes_moved / spec.bandwidth else "memory"
    pct = 100 * roof_s / s
    line = (f"{label:<34} {ms*1e3:9.1f} us | {gbps:8.1f} GB/s | {tflops:7.2f} TFLOPS | "
            f"AI {intensity:8.2f} | {bound}-bound | {pct:5.1f}% of roof")
    if baseline_ms is not None:
        line += f" | {baseline_ms / ms:4.2f}x vs torch"
    print(line)
