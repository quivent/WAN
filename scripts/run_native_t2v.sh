#!/usr/bin/env bash
set -euo pipefail

WAN_NATIVE_REPO="${WAN_NATIVE_REPO:-/opt/Wan2.2}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-/models/Wan2.2-T2V-A14B}"
GPUS="${GPUS:-8}"
SIZE="${SIZE:-1280*720}"
PROMPT="${PROMPT:?set PROMPT}"

cd "$WAN_NATIVE_REPO"
torchrun --nproc_per_node="$GPUS" generate.py \
  --task t2v-A14B \
  --size "$SIZE" \
  --ckpt_dir "$WAN_MODEL_DIR" \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size "$GPUS" \
  --offload_model True \
  --convert_model_dtype \
  --prompt "$PROMPT"
