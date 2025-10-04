#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$SCRIPT_DIR"
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1

