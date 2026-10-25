#!/usr/bin/env bash
# Build the Linux test image (CPU PyTorch + Triton interpreter). Re-run only if
# docker/Dockerfile changes; your code is mounted in, not copied.
# Usage:  poe setup
set -e
cd "$(dirname "$0")/.."
source scripts/lib.sh

docker info >/dev/null 2>&1 || { echo "Docker isn't running. Start Docker Desktop and retry." >&2; exit 1; }
docker build -t "$IMAGE" docker
echo ""
echo "Ready. Run tests with: uv run poe test 1"
