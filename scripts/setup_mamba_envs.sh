#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA_BIN="${MAMBA_BIN:-mamba}"
MAIN_ENV="$ROOT_DIR/.mamba/envs/sg-nav"
VLLM_ENV="$ROOT_DIR/.mamba/envs/sg-nav-vllm"

create_or_update_env() {
  local prefix="$1"
  local yaml="$2"
  if [[ -d "$prefix" ]]; then
    "$MAMBA_BIN" env update -p "$prefix" -f "$yaml" --prune
  else
    "$MAMBA_BIN" env create -p "$prefix" -f "$yaml"
  fi
}

cd "$ROOT_DIR"

create_or_update_env "$MAIN_ENV" "$ROOT_DIR/envs/sg-nav.yml"
"$MAIN_ENV/bin/pip" install -r "$ROOT_DIR/envs/sg-nav-pip.txt"

create_or_update_env "$VLLM_ENV" "$ROOT_DIR/envs/sg-nav-vllm.yml"
"$VLLM_ENV/bin/pip" install -r "$ROOT_DIR/envs/sg-nav-vllm-pip.txt"

"$MAIN_ENV/bin/python" "$ROOT_DIR/check_setup.py" || true

echo
echo "Mamba environments are ready:"
echo "  $MAIN_ENV"
echo "  $VLLM_ENV"
echo
echo "Start vLLM with:"
echo "  ./run_vllm.sh"
echo
echo "Run SG-Nav with:"
echo "  ./run_sg_nav.sh --split_l 0 --split_r 1 --num_episodes 1 --debug_sgnav"
