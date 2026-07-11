#!/usr/bin/env bash
# One-command runner for the scripted drone-arm gate fly-through demo.
#
# Default scene requested by the user:
#   drone-arm horizontal start: (0, 0)
#   gate horizontal center:     (-1, 1)
#   flight/gate center height:  1.05 m

set -euo pipefail

MODE="${1:-gui}"
ENV_NAME="${ENV_NAME:-so-arm101-isaac}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DRONE_ARM_URDF_PATH="${DRONE_ARM_URDF_PATH:-$HOME/blander/drone_with_arm.urdf}"

GATE_X="${GATE_X:--1.0}"
GATE_Y="${GATE_Y:-1.0}"
FLIGHT_Z="${FLIGHT_Z:-1.05}"
GATE_WIDTH="${GATE_WIDTH:-1.2}"
GATE_HEIGHT="${GATE_HEIGHT:-1.0}"
BAR_SIZE="${BAR_SIZE:-0.08}"
PASS_DISTANCE="${PASS_DISTANCE:-1.2}"
YAW_OFFSET_DEG="${YAW_OFFSET_DEG:-0.0}"
MAX_FRAMES="${MAX_FRAMES:-720}"

if [[ ! -f "${DRONE_ARM_URDF_PATH}" ]]; then
  echo "[run-gate] missing URDF: ${DRONE_ARM_URDF_PATH}" >&2
  echo "[run-gate] set DRONE_ARM_URDF_PATH=/path/to/blander/drone_with_arm.urdf" >&2
  exit 2
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
fi

cd "${PROJECT_DIR}"
export DRONE_ARM_URDF_PATH

SCRIPT_ARGS=(
  --urdf-path "${DRONE_ARM_URDF_PATH}"
  --gate-x "${GATE_X}"
  --gate-y "${GATE_Y}"
  --flight-z "${FLIGHT_Z}"
  --gate-width "${GATE_WIDTH}"
  --gate-height "${GATE_HEIGHT}"
  --bar-size "${BAR_SIZE}"
  --pass-distance "${PASS_DISTANCE}"
  --yaw-offset-deg "${YAW_OFFSET_DEG}"
  --max-frames "${MAX_FRAMES}"
)

case "${MODE}" in
  gui)
    echo "[run-gate] GUI loop: gate=(${GATE_X}, ${GATE_Y}, ${FLIGHT_Z})"
    python scripted_gate_flythrough.py "${SCRIPT_ARGS[@]}" --loop
    ;;
  headless|once)
    echo "[run-gate] headless once: gate=(${GATE_X}, ${GATE_Y}, ${FLIGHT_Z})"
    python scripted_gate_flythrough.py "${SCRIPT_ARGS[@]}" --headless
    ;;
  *)
    echo "[run-gate] unknown mode: ${MODE}" >&2
    echo "[run-gate] valid modes: gui, headless, once" >&2
    exit 2
    ;;
esac
