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
  query-model)
    shift || true
    "$ROOT/.venv/bin/pip" install -q -r signals/query_model/requirements.txt
    # Faster HF Hub downloads (requires hf_transfer in requirements). Safe default;
    # does not affect any already-running download process.
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    exec "$ROOT/.venv/bin/python" "$ROOT/signals/query_model/build.py" "$@"
    ;;
  query-model-analyze)
    shift || true
    "$ROOT/.venv/bin/pip" install -q -r signals/query_model/requirements.txt
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    exec "$ROOT/.venv/bin/python" "$ROOT/signals/query_model/analyze.py" "$@"
    ;;
  query-model-restore)
    shift || true
    exec "$ROOT/.venv/bin/python" "$ROOT/signals/query_model/restore.py" "$@"
    ;;
  route)
    # Rule-based router (router/rule_based). Learn-weight later: router/learn_weight/
    shift || true
    "$ROOT/.venv/bin/pip" install -q -r signals/query_model/requirements.txt
    exec "$ROOT/.venv/bin/python" "$ROOT/router/rule_based/cli.py" "$@"
    ;;
  *)
    echo "usage: ./run.sh [--smoke|corpus|complexity|query-model|query-model-analyze|query-model-restore|route] [args...]"
    exit 1
    ;;
esac
