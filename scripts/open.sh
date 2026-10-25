#!/usr/bin/env bash
# Open an exercise in VS Code or Cursor as tabs README > kernel > hints (docs as markdown previews),
# with kernel.py focused, ready to edit.
# Usage:  poe open 3              (or 03, or ex03_rmsnorm)
# Uses $EDITOR_CLI if set, else `cursor`, else `code`.
set -e
cd "$(dirname "$0")/.."
source scripts/lib.sh

dir=$(resolve_ex "${1:?usage: poe open <exercise number or name>}")

editor=$(editor_cli)
[ -n "$editor" ] || { echo "No 'cursor' or 'code' command. Install it from the editor's Command Palette." >&2; exit 1; }

# .vscode/settings.json maps exercise README.md and hints.md to the markdown preview editor.
"$editor" -r . "$dir/README.md" "$dir/kernel.py" "$dir/hints.md"
"$editor" -r "$dir/kernel.py"  # refocus the kernel tab without reordering
