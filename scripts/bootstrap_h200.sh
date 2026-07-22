#!/usr/bin/env bash
set -euo pipefail

WAN_HOME="${WAN_HOME:-$PWD}"
WAN_NATIVE_REPO="${WAN_NATIVE_REPO:-/opt/Wan2.2}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-/models/Wan2.2-T2V-A14B}"
WAN_OUTPUT_DIR="${WAN_OUTPUT_DIR:-/runs/wan/outputs}"
WAN_STATE_DIR="${WAN_STATE_DIR:-/runs/wan/.wand}"
PYTHON="${PYTHON:-python3.11}"

mkdir -p "$WAN_OUTPUT_DIR" "$WAN_STATE_DIR" /runs/wan/logs /models

if [[ ! -d "$WAN_NATIVE_REPO/.git" ]]; then
  git clone https://github.com/Wan-Video/Wan2.2.git "$WAN_NATIVE_REPO"
fi

cd "$WAN_HOME"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .

echo "WAN_HOME=$WAN_HOME"
echo "WAN_NATIVE_REPO=$WAN_NATIVE_REPO"
echo "WAN_MODEL_DIR=$WAN_MODEL_DIR"
echo "WAN_OUTPUT_DIR=$WAN_OUTPUT_DIR"
echo "WAN_STATE_DIR=$WAN_STATE_DIR"
echo "Run: wan doctor"
