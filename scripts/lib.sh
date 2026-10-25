# Shared helpers, sourced by the other scripts.

# Turn "3", "03" or "ex03_rmsnorm" into the exercise folder name, or exit with the list.
resolve_ex() {
  local arg="$1" dir
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    dir=$(ls -d ex$(printf '%02d' "$((10#$arg))")_*/ 2>/dev/null | head -1)
  else
    dir="$arg"
  fi
  dir="${dir%/}"
  [ -d "$dir" ] || { echo "no exercise matching '$arg'" >&2; ls -d ex*/ >&2; exit 1; }
  echo "$dir"
}

IMAGE=kernel-lab-cpu

# The editor's shell command: $EDITOR_CLI if set, else the editor this terminal runs in,
# else whichever of `cursor` or `code` is installed. Empty if none.
editor_cli() {
  if [ -n "$EDITOR_CLI" ]; then echo "$EDITOR_CLI"
  elif [[ "$TERM_PROGRAM" = vscode && "$VSCODE_GIT_ASKPASS_NODE" == *Cursor* ]] && command -v cursor >/dev/null; then echo cursor
  elif [ "$TERM_PROGRAM" = vscode ] && command -v code >/dev/null; then echo code
  else command -v cursor || command -v code || true
  fi
}
