#!/usr/bin/env bash
set -euo pipefail

WAN_HOME="${WAN_HOME:-/opt/WAN}"
export WAN_NATIVE_REPO="${WAN_NATIVE_REPO:-/opt/Wan2.2}"
export WAN_MODEL_DIR="${WAN_MODEL_DIR:-/models/Wan2.2-T2V-A14B}"
export WAN_OUTPUT_DIR="${WAN_OUTPUT_DIR:-/runs/wan/outputs}"
export WAN_STATE_DIR="${WAN_STATE_DIR:-/runs/wan/.wand}"
POLL_SECONDS="${POLL_SECONDS:-10}"

cd "$WAN_HOME"
exec .venv/bin/wan worker --state-dir "$WAN_STATE_DIR" --poll "$POLL_SECONDS"
