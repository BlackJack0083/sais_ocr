#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <session_name> <workdir> <log_file> <command...>" >&2
  exit 1
fi

session_name="$1"
workdir="$2"
log_file="$3"
shift 3

mkdir -p "$(dirname "$log_file")"

if tmux has-session -t "$session_name" 2>/dev/null; then
  tmux kill-session -t "$session_name"
fi

command_string="$*"
tmux new-session -d -s "$session_name" "bash -lc 'cd \"$workdir\" && { echo \"[tmux-session] $session_name\"; echo \"[started-at] \$(date \"+%F %T\")\"; echo \"[command] $command_string\"; $command_string; rc=\$?; echo \"[finished-at] \$(date \"+%F %T\")\"; echo \"[exit-code] \$rc\"; exit \$rc; } 2>&1 | tee -a \"$log_file\"'"
echo "$session_name"
