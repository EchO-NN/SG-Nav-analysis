#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/echo/SG-Nav"
cd "$ROOT_DIR"

mkdir -p \
  "$ROOT_DIR/data/results" \
  "$ROOT_DIR/data/debug_sgnav" \
  "$ROOT_DIR/data/visualization" \
  "$ROOT_DIR/.cache/matplotlib" \
  "$ROOT_DIR/.cache/torch"

case "${1:-bash}" in
  bash|sh)
    exec "$@"
    ;;
  check)
    shift
    exec "$ROOT_DIR/.mamba/envs/sg-nav/bin/python" "$ROOT_DIR/check_setup.py" "$@"
    ;;
  vllm)
    shift
    export VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
    exec "$ROOT_DIR/run_vllm.sh" "$@"
    ;;
  sg-nav)
    shift
    exec "$ROOT_DIR/run_sg_nav.sh" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
