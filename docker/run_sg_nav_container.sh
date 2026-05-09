#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-sgnav-pro6000:latest}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/docker_outputs}"

mkdir -p "$OUT_DIR/results" "$OUT_DIR/debug_sgnav" "$OUT_DIR/visualization"

docker run \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --rm \
  -it \
  --network host \
  -e VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}" \
  -v "$OUT_DIR/results:/home/echo/SG-Nav/data/results" \
  -v "$OUT_DIR/debug_sgnav:/home/echo/SG-Nav/data/debug_sgnav" \
  -v "$OUT_DIR/visualization:/home/echo/SG-Nav/data/visualization" \
  "$IMAGE_NAME" \
  sg-nav "$@"
