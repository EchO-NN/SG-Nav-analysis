#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SG_NAV_ENV="$ROOT_DIR/.mamba/envs/sg-nav"

export QT_X11_NO_MITSHM=1
exec "$SG_NAV_ENV/bin/python" "$ROOT_DIR/tools/realtime_monitor_viewer.py" "$@"
