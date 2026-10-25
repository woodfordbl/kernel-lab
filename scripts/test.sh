#!/usr/bin/env bash
# Run an exercise's tests in the Linux image on Triton's CPU interpreter (correctness only).
# Usage:  poe test 3              (or 03, or ex03_rmsnorm; extra args go to pytest)
#         poe test                (every exercise)
set -e
cd "$(dirname "$0")/.."
source scripts/lib.sh

target=()
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  target=("$(resolve_ex "$1")")
  shift
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "No test image yet. Run: uv run poe setup" >&2; exit 1; }
docker run --rm -t -v "$PWD:/lab" "$IMAGE" pytest -p no:cacheprovider "${target[@]}" "$@"
