#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-sgnav-pro6000:latest}"
CUDA_IMAGE="${CUDA_IMAGE:-nvcr.io/nvidia/cuda:12.8.1-base-ubuntu22.04}"
UID_ARG="${UID_ARG:-$(id -u)}"
GID_ARG="${GID_ARG:-$(id -g)}"
read -r -a DOCKER_CMD <<< "${DOCKER_BIN:-docker}"
BUILD_ARGS=()
if [[ -n "${BUILD_NETWORK:-}" ]]; then
  BUILD_ARGS+=(--network "$BUILD_NETWORK")
fi

"${DOCKER_CMD[@]}" build \
  "${BUILD_ARGS[@]}" \
  --build-arg CUDA_IMAGE="$CUDA_IMAGE" \
  --build-arg UID="$UID_ARG" \
  --build-arg GID="$GID_ARG" \
  -t "$IMAGE_NAME" \
  "$ROOT_DIR"
