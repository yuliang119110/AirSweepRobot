#!/bin/bash
# launch_gui.sh - 在 /home/dgx/isaac_sim_test 下以 GUI 模式运行 Isaac Sim 6.0.1
set -e

IMAGE="isaac-sim-torch:6.0.1"
CONTAINER="isaac_sim_test_gui"
WORKSPACE="/home/dgx/isaac_sim_test"

echo "========================================================="
echo "[GUI] Isaac Sim 6.0.1 GUI 仿真测试"
echo "[GUI] Display: :0"
echo "[GUI] Workspace: $WORKSPACE"
echo "========================================================="

# 清理旧容器
docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true

# 授权 X11
export DISPLAY=:0
xhost +local:docker 2>/dev/null || true

# 以 GUI 模式启动 SimulationApp (headless=False,自带窗口)
# 覆盖 Docker entrypoint 绕过 runheadless.sh,让 SimulationApp 自行启动 kit
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
  -v $WORKSPACE:/workspace \
  -w /workspace \
  $IMAGE \
  /workspace/sim_test.py

echo ""
echo "[GUI] 容器已启动,Isaac Sim 窗口应显示在物理显示器 :0 上"
echo "[GUI] 查看日志: docker logs -f $CONTAINER"
echo "[GUI] 停止: docker stop $CONTAINER"
