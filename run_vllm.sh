#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ENV="$ROOT_DIR/.mamba/envs/sg-nav-vllm"

if [[ $# -gt 0 && "$1" != -* ]]; then
  MODEL="$1"
  shift
else
  MODEL="${VLLM_HF_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
fi

HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${VLLM_MODEL:-qwen3-vl-8b-instruct}}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-{\"image\":1}}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-0}"

EXTRA_ARGS=()
if [[ "$ENFORCE_EAGER" != "0" && "$ENFORCE_EAGER" != "false" && "$ENFORCE_EAGER" != "False" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "$ENABLE_PREFIX_CACHING" == "0" || "$ENABLE_PREFIX_CACHING" == "false" || "$ENABLE_PREFIX_CACHING" == "False" ]]; then
  EXTRA_ARGS+=(--no-enable-prefix-caching)
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export LD_LIBRARY_PATH="$VLLM_ENV/lib:${LD_LIBRARY_PATH:-}"

exec "$VLLM_ENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
  "${EXTRA_ARGS[@]}" \
  "$@"
