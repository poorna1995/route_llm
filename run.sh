#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -q -r datasets/requirements.txt
fi

cmd="${1:-corpus}"
case "$cmd" in
  --smoke)
    exec "$ROOT/.venv/bin/python" "$ROOT/datasets/build.py" --config datasets/config.yaml --smoke
    ;;
  corpus)
    shift || true
    SMOKE=()
    if [[ "${1:-}" == "--smoke" ]]; then
      SMOKE=(--smoke)
      shift
    fi
    CONFIG="${1:-datasets/config.yaml}"
    exec "$ROOT/.venv/bin/python" "$ROOT/datasets/build.py" --config "$CONFIG" "${SMOKE[@]}"
    ;;
  complexity)
    shift || true
    "$ROOT/.venv/bin/pip" install -q -r signals/query/requirements.txt
    exec "$ROOT/.venv/bin/python" "$ROOT/signals/query/build.py" "$@"
    ;;
  *)
    echo "usage: ./run.sh [--smoke|corpus|complexity] [args...]"
    exit 1
    ;;
esac
