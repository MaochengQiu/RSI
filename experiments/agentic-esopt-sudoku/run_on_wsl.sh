#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
cache="$PWD/.cache"
mkdir -p "$cache/tmp" "$cache/hf"
export TMPDIR="$cache/tmp" TMP="$cache/tmp" TEMP="$cache/tmp"
export XDG_CACHE_HOME="$cache" HF_HOME="$cache/hf" CUDA_CACHE_PATH="$cache/cuda"
export TORCH_EXTENSIONS_DIR="$cache/torch-extensions" TRITON_CACHE_DIR="$cache/triton"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
case "${1:-}" in
  observable-es|direction-transfer)
    case "$1" in
      observable-es) script=run_observable_es.py ;;
      direction-transfer) script=run_direction_transfer.py ;;
    esac
    shift
    export PYTHONPATH="$PWD/.runtime"
    exec ~/.venvs/rsi-alfworld/bin/python -u "$script" "$@"
    ;;
esac
exec ~/.venvs/rsi-alfworld/bin/python -u run_local.py "$@"
