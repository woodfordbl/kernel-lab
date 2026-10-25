#!/usr/bin/env bash
# Quick push: stages all changes, commits with a message, and pushes.
# Usage:  poe push "your message"
#         poe push              (defaults to "wip")
set -e
cd "$(dirname "$0")/.."
msg="${1:-wip}"
git add -A
git commit -m "$msg" || echo "nothing to commit"
git push
echo ""
echo "Pushed. In colab.ipynb, run 'Test and benchmark' (it pulls first)."
