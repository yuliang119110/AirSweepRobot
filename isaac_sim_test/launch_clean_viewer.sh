#!/bin/bash
# launch_clean_viewer.sh - SO-ARM101 清洁场景 GUI 可视化
# 纯 SimulationApp + USD API，绕过 Fabric（根因修复）
set -e

IMAGE="isaac-sim-torch:6.0.1"
CONTAINER="isaac_clean_viewer"
WORKSPACE="/home/dgx/air/isaac_sim_test"
MODEL_TEST="/home/dgx/model_test"

echo "========================================================="
echo "[Clean] SO-ARM101 清洁场景可视化 (纯 SimulationApp)"
echo "[Clean] Display: :0"
echo "[Clean] Image:   $IMAGE"
echo "========================================================="

docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true

export DISPLAY=:0
xhost +local:docker 2>/dev/null || true

# Script selection: clean_scene.py (new) or clean_scene_viewer.py (old)
SCRIPT="${SCRIPT:-clean_scene.py}"
EXTRA_ARGS="${@:-}"

docker run -d \
  --name $CONTAINER \
  --entrypoint "/isaac-sim/python.sh" \
  --gpus all \
  --ipc=host \
  --network=host \
  -e "ACCEPT_EULA=Y" \
  -e "DISPLAY=:0" \
  -e "XAUTHORITY=/root/.Xauthority" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$WORKSPACE:/workspace" \
  -v "$MODEL_TEST:/model_test" \
  -w /workspace \
  $IMAGE \
  /workspace/${SCRIPT} ${EXTRA_ARGS}

echo ""
echo "[Clean] 容器已启动,Isaac Sim 窗口应显示在物理显示器 :0 上"
echo "[Clean] 查看日志: docker logs -f $CONTAINER"
echo "[Clean] 停止: docker stop $CONTAINER"
