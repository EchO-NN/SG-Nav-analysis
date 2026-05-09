#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-sgnav-pro6000:latest}"
OUT="${OUT:-$ROOT_DIR/dist/sgnav-pro6000-image.tar.gz}"

mkdir -p "$(dirname "$OUT")"
docker save "$IMAGE_NAME" | gzip -1 > "$OUT"
ls -lh "$OUT"
