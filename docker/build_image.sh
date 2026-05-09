#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-sgnav-pro6000:latest}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04}"
UID_ARG="${UID_ARG:-$(id -u)}"
GID_ARG="${GID_ARG:-$(id -g)}"

docker build \
  --build-arg CUDA_IMAGE="$CUDA_IMAGE" \
  --build-arg UID="$UID_ARG" \
  --build-arg GID="$GID_ARG" \
  -t "$IMAGE_NAME" \
  "$ROOT_DIR"
