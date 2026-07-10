#!/bin/bash
set -e

# ============================================================
# LocateAnything launcher script
# All optimization options are controlled by environment variables.
# Override any of them before running this script, e.g.:
#   LA_FP8=1 ./run.sh
#   LA_COMPILE=0 ./run.sh
# ============================================================

# --- Conda environment ---
CONDA_BASE="/home/dgx/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate locate-anything

# --- HuggingFace mirror (direct HF is blocked in this environment) ---
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# --- Model cache (avoids root-owned ~/.cache/huggingface) ---
export HF_HOME="${HF_HOME:-/home/dgx/air/object-detection/hf_cache}"

# --- L0: GPU backend ---
export LA_TF32="${LA_TF32:-1}"
export LA_CUDNN_BENCH="${LA_CUDNN_BENCH:-1}"

# --- L1: torch.compile ---
export LA_COMPILE="${LA_COMPILE:-1}"
export LA_COMPILE_MODE="${LA_COMPILE_MODE:-reduce-overhead}"

# --- L2: FP8 quantization ---
export LA_FP8="${LA_FP8:-0}"

# --- Warmup on startup ---
export LA_WARMUP="${LA_WARMUP:-1}"

# --- Generation mode ---
export LA_GEN_MODE="${LA_GEN_MODE:-hybrid}"

# --- Server ---
export LA_PORT="${LA_PORT:-7860}"

cd "$(dirname "$0")"
exec python app.py
