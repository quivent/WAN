#!/usr/bin/env bash
#SBATCH --job-name=wan-worker-h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export WAN_HOME="${WAN_HOME:-$PWD}"
export WAN_NATIVE_REPO="${WAN_NATIVE_REPO:-/opt/Wan2.2}"
export WAN_MODEL_DIR="${WAN_MODEL_DIR:-/models/Wan2.2-T2V-A14B}"
export WAN_OUTPUT_DIR="${WAN_OUTPUT_DIR:-/runs/wan/outputs}"
export WAN_STATE_DIR="${WAN_STATE_DIR:-/runs/wan/.wand}"

cd "$WAN_HOME"
.venv/bin/wan worker --state-dir "$WAN_STATE_DIR" --poll 10
