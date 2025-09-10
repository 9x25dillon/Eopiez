#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export JULIA_PROJECT="$SCRIPT_DIR"
exec julia --color=yes "$SCRIPT_DIR/src/qvnm_server.jl"

