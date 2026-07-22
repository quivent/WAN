#!/usr/bin/env bash
#SBATCH --job-name=wan-t2v
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export WAN_NATIVE_REPO="${WAN_NATIVE_REPO:-/opt/Wan2.2}"
export WAN_MODEL_DIR="${WAN_MODEL_DIR:-/models/Wan2.2-T2V-A14B}"
export PROMPT="${PROMPT:?set PROMPT}"

cd "$WAN_NATIVE_REPO"
torchrun --nproc_per_node=8 generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir "$WAN_MODEL_DIR" \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --offload_model True \
  --convert_model_dtype \
  --prompt "$PROMPT"
