#!/usr/bin/env bash
# Debug checklist for the scripted drone-arm gate fly-through demo.

set -euo pipefail

MODE="${1:-check}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DRONE_ARM_URDF_PATH="${DRONE_ARM_URDF_PATH:-$HOME/blander/drone_with_arm.urdf}"
MESH_DIR="${MESH_DIR:-$(dirname "${DRONE_ARM_URDF_PATH}")/meshes}"

GATE_X="${GATE_X:--1.0}"
GATE_Y="${GATE_Y:-1.0}"
FLIGHT_Z="${FLIGHT_Z:-1.05}"

fail() {
  echo "[debug-gate] FAIL: $*" >&2
  exit 2
}

warn() {
  echo "[debug-gate] WARN: $*" >&2
}

ok() {
  echo "[debug-gate] OK: $*"
}

cd "${PROJECT_DIR}"

echo "[debug-gate] project: ${PROJECT_DIR}"
echo "[debug-gate] urdf:    ${DRONE_ARM_URDF_PATH}"
echo "[debug-gate] meshes:  ${MESH_DIR}"
echo "[debug-gate] gate:    (${GATE_X}, ${GATE_Y}, ${FLIGHT_Z})"

[[ -f scripted_gate_flythrough.py ]] || fail "scripted_gate_flythrough.py not found"
[[ -f run_gate_demo.sh ]] || warn "run_gate_demo.sh not found"
[[ -f "${DRONE_ARM_URDF_PATH}" ]] || fail "URDF not found. Set DRONE_ARM_URDF_PATH."
[[ -d "${MESH_DIR}" ]] || fail "mesh directory not found: ${MESH_DIR}"

COLLISION_COUNT="$(grep -c "<collision" "${DRONE_ARM_URDF_PATH}" || true)"
if [[ "${COLLISION_COUNT}" -gt 0 ]]; then
  ok "URDF collision tags: ${COLLISION_COUNT}"
else
  warn "URDF has no <collision> tags; visual demo still works, physics collision will not be reliable"
fi

REQUIRED_MESHES=(
  drone_body.stl
  drone_body_collision.stl
)

for mesh in "${REQUIRED_MESHES[@]}"; do
  if [[ -f "${MESH_DIR}/${mesh}" ]]; then
    ok "mesh exists: ${mesh}"
  else
    warn "mesh not found: ${MESH_DIR}/${mesh}"
  fi
done

if command -v python >/dev/null 2>&1; then
  python -m py_compile scripted_gate_flythrough.py
  ok "Python syntax check passed"
else
  warn "python command not found; skip syntax check"
fi

if command -v bash >/dev/null 2>&1; then
  bash -n run_gate_demo.sh
  bash -n launch_conda_aerial_gate.sh
  ok "bash syntax check passed"
fi

case "${MODE}" in
  check)
    echo "[debug-gate] check complete"
    ;;
  smoke)
    echo "[debug-gate] running headless smoke test"
    DRONE_ARM_URDF_PATH="${DRONE_ARM_URDF_PATH}" \
    GATE_X="${GATE_X}" \
    GATE_Y="${GATE_Y}" \
    FLIGHT_Z="${FLIGHT_Z}" \
    MAX_FRAMES="${MAX_FRAMES:-180}" \
    ./run_gate_demo.sh headless
    ;;
  *)
    fail "unknown mode: ${MODE}. valid modes: check, smoke"
    ;;
esac
