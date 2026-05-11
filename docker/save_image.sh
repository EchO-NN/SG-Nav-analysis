#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-sgnav-pro6000:latest}"
OUT="${OUT:-$ROOT_DIR/dist/sgnav-pro6000-image.tar.gz}"
read -r -a DOCKER_CMD <<< "${DOCKER_BIN:-docker}"

mkdir -p "$(dirname "$OUT")"
"${DOCKER_CMD[@]}" save "$IMAGE_NAME" | gzip -1 > "$OUT"
ls -lh "$OUT"
