#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
  # Dependencies are checked/installed separately; keeping this step offline
  # avoids contacting a broken package mirror during editable installation.
  python3 -m pip install --no-build-isolation --no-deps -e "$ROOT_DIR"
else
  python3 -m pip install --user --no-build-isolation --no-deps -e "$ROOT_DIR"
fi
echo "object-modeling installed from $ROOT_DIR"
