#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="${OBJECT_MODELING_CONFIG:-$ROOT_DIR/configs/d405_charuco.yaml}"
DATASET="${1:?用法: $0 DATASET_DIR [额外参数]}"
shift
exec python3 -m object_modeling.cli build --dataset "$DATASET" --config "$CONFIG" "$@"
