"""Shared helpers for kernel-lab: device/dtype selection, roofline numbers, benchmarking."""
from .env import DEVICE, DTYPE, ON_GPU
from .roofline import gpu_spec, report
from .bench import time_ms

__all__ = ["DEVICE", "DTYPE", "ON_GPU", "gpu_spec", "report", "time_ms"]
