# Linux image for running the tests on a Mac: CPU-only PyTorch plus Triton, whose
# interpreter (TRITON_INTERPRET=1, set by common/env.py) runs the kernels on the CPU.
# The repo is mounted at /lab at run time, so edits never need a rebuild.
FROM python:3.12-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "triton>=3.0" numpy pytest
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /lab
