#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SG_NAV_ENV="$ROOT_DIR/.mamba/envs/sg-nav"
TORCH_LIB_DIR="$SG_NAV_ENV/lib/python3.9/site-packages/torch/lib"
CUDA12_RUNTIME_DIR="/usr/local/lib/ollama/cuda_v12"
PORTABLE_CUDA12_RUNTIME_DIR="$ROOT_DIR/.cuda_v12"

export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA="$ROOT_DIR/.cache/nltk_data"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONWARNINGS=ignore
export GYM_DISABLE_WARNINGS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export PATH="$SG_NAV_ENV/bin:${PATH:-}"
export PYTHONPATH="$ROOT_DIR/habitat-lab:$ROOT_DIR/GroundingDINO:$ROOT_DIR/GLIP:$ROOT_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$SG_NAV_ENV/lib:${LD_LIBRARY_PATH:-}"
if [[ -d "$PORTABLE_CUDA12_RUNTIME_DIR" ]]; then
  export LD_LIBRARY_PATH="$PORTABLE_CUDA12_RUNTIME_DIR:$LD_LIBRARY_PATH"
fi
if [[ -d "$CUDA12_RUNTIME_DIR" ]]; then
  export LD_LIBRARY_PATH="$CUDA12_RUNTIME_DIR:$LD_LIBRARY_PATH"
fi

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export VLLM_MODEL="${VLLM_MODEL:-${VLLM_SERVED_MODEL_NAME:-qwen3-vl-8b-instruct}}"
export VLLM_TIMEOUT="${VLLM_TIMEOUT:-120}"
export VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-256}"
export VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-0}"
export VLLM_TOP_P="${VLLM_TOP_P:-1.0}"
export VLLM_SEED="${VLLM_SEED:-0}"
export VLLM_DISABLE_THINKING="${VLLM_DISABLE_THINKING:-1}"

exec "$SG_NAV_ENV/bin/python" "$ROOT_DIR/SG_Nav.py" "$@"
