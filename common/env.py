"""Pick where tests run.

On a CUDA GPU: real kernels in fp16.
Without one: Triton's interpreter (TRITON_INTERPRET=1) runs kernels on the CPU in fp32.
The interpreter is slow and only checks correctness, never speed. conftest.py sets the
environment variable before Triton is imported, so you never need to set it yourself.
"""
import os

import torch

ON_GPU = torch.cuda.is_available()
if not ON_GPU:
    os.environ.setdefault("TRITON_INTERPRET", "1")

DEVICE = "cuda" if ON_GPU else "cpu"
DTYPE = torch.float16 if ON_GPU else torch.float32
