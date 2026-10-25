#!/usr/bin/env bash
# Start an AI coach for an exercise. It follows the coaching rules in AGENTS.md: hints and
# questions, not solutions.
# Usage:  poe agent-help 3            Claude Code in the editor's right sidebar, if the
#                                     extension is installed; else the claude CLI here
#         poe agent-help 3 claude     the claude CLI in this terminal
#         poe agent-help 3 codex      the codex CLI in this terminal
set -e
cd "$(dirname "$0")/.."
source scripts/lib.sh

dir=$(resolve_ex "${1:?usage: poe agent-help <exercise number or name> [claude|codex]}")
agent="${2:-}"

prompt="Coach me through $dir. Follow the coaching rules in AGENTS.md. Read $dir/README.md and my $dir/kernel.py, then ask me where I'm stuck."

if [ -z "$agent" ]; then
  editor=$(editor_cli)
  if [ -n "$editor" ] && "$editor" --list-extensions 2>/dev/null | grep -qi '^anthropic.claude-code$'; then
    case "$(basename "$editor")" in cursor) scheme=cursor ;; *) scheme=vscode ;; esac
    encoded=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$prompt")
    "$editor" -r . "$dir/kernel.py"
    open "$scheme://anthropic.claude-code/open?prompt=$encoded" 2>/dev/null \
      || xdg-open "$scheme://anthropic.claude-code/open?prompt=$encoded"
    echo "Opened Claude Code with the coaching prompt filled in. Press Enter in its panel to start."
    exit 0
  fi
  agent=claude
fi

command -v "$agent" >/dev/null || { echo "'$agent' isn't installed." >&2; exit 1; }
exec "$agent" "$prompt"
