#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-sgnav-pro6000:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-sgnav-vllm}"

docker run \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --rm \
  -it \
  --name "$CONTAINER_NAME" \
  -p 8000:8000 \
  -e VLLM_HOST=0.0.0.0 \
  "$IMAGE_NAME" \
  vllm "$@"
