#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="${OBJECT_MODELING_CONFIG:-$ROOT_DIR/configs/d405_charuco.yaml}"
OUTPUT="${1:?Usage: bash $0 DATASET_DIR [capture options]}"
shift
# Source your ROS distribution before running this script.
exec python3 -m object_modeling.cli capture --config "$CONFIG" --output "$OUTPUT" --show "$@"
