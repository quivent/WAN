#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Wan-AI/Wan2.2-T2V-A14B}"
LOCAL_DIR="${2:-/models/Wan2.2-T2V-A14B}"

huggingface-cli download "$MODEL" --local-dir "$LOCAL_DIR"
