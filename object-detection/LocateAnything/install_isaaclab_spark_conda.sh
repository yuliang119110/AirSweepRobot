#!/usr/bin/env bash
set -euo pipefail

# Install Isaac Sim 5.1 + Isaac Lab into an isolated conda env on NVIDIA Spark.
# Usage:
#   bash install_isaaclab_spark_conda.sh
#
# Optional env vars:
#   ENV_NAME=isaaclab
#   WORKDIR=$HOME/robot-act/sim
#   ISAACLAB_BRANCH=main
#   INSTALL_FRAMEWORKS=none   # none | all | rl_games | rsl_rl | sb3 | skrl | robomimic

ENV_NAME="${ENV_NAME:-isaaclab}"
WORKDIR="${WORKDIR:-$HOME/robot-act/sim}"
ISAACLAB_BRANCH="${ISAACLAB_BRANCH:-main}"
INSTALL_FRAMEWORKS="${INSTALL_FRAMEWORKS:-none}"

echo "== NVIDIA Spark Isaac Lab conda installer =="
echo "ENV_NAME=${ENV_NAME}"
echo "WORKDIR=${WORKDIR}"
echo "ISAACLAB_BRANCH=${ISAACLAB_BRANCH}"
echo "INSTALL_FRAMEWORKS=${INSTALL_FRAMEWORKS}"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found. Install Miniforge/Miniconda for this Spark architecture first." >&2
    exit 1
  fi
fi

ARCH="$(uname -m)"
echo "Detected architecture: ${ARCH}"

echo "== System check =="
uname -a || true
nvidia-smi || true
conda --version

echo "== Create / activate conda env =="
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists; reusing it."
else
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

echo "== Install Isaac Sim 5.1 pip packages =="
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

echo "== Install CUDA-enabled PyTorch for architecture =="
if [ "${ARCH}" = "aarch64" ] || [ "${ARCH}" = "arm64" ]; then
  python -m pip install -U torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130
  if [ -f /lib/aarch64-linux-gnu/libgomp.so.1 ]; then
    export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
    mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
    cat > "${CONDA_PREFIX}/etc/conda/activate.d/isaaclab_aarch64_libgomp.sh" <<'EOF'
unset LD_PRELOAD
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
EOF
  fi
else
  python -m pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
fi

echo "== Install system build dependencies if sudo is available =="
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y cmake build-essential git git-lfs
else
  echo "WARN: sudo not found; skipping apt dependencies. Ensure cmake/build-essential/git/git-lfs are installed."
fi

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

echo "== Clone / update IsaacLab =="
if [ -d IsaacLab/.git ]; then
  cd IsaacLab
  git fetch origin "${ISAACLAB_BRANCH}"
  git checkout "${ISAACLAB_BRANCH}"
  git pull --ff-only origin "${ISAACLAB_BRANCH}"
else
  git clone https://github.com/isaac-sim/IsaacLab.git --branch "${ISAACLAB_BRANCH}"
  cd IsaacLab
fi

echo "== Install Isaac Lab extensions =="
./isaaclab.sh --install "${INSTALL_FRAMEWORKS}"

echo "== Verify Python imports =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
from isaacsim import SimulationApp
print("isaacsim SimulationApp import: OK")
import isaaclab
print("isaaclab import: OK")
PY

echo "== Installation finished =="
echo "Activate later with:"
echo "  conda activate ${ENV_NAME}"
echo "Run a headless smoke test with:"
echo "  cd ${WORKDIR}/IsaacLab"
echo "  OMNI_KIT_ACCEPT_EULA=YES python scripts/tutorials/00_sim/create_empty.py --headless"
