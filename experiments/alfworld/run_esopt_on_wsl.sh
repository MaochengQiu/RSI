#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
cache="$HOME/rsi-data/alfworld/esopt-cache"
mkdir -p "$cache/tmp" "$cache/hf"
export TMPDIR="$cache/tmp" TMP="$cache/tmp" TEMP="$cache/tmp"
export HF_HOME="$cache/hf" XDG_CACHE_HOME="$cache" CUDA_CACHE_PATH="$cache/cuda"
export TRITON_CACHE_DIR="$cache/triton" TORCH_EXTENSIONS_DIR="$cache/torch-extensions"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
exec ~/.venvs/rsi-alfworld/bin/python -u -B run_esopt.py "$@"
