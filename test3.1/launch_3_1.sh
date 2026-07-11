#!/bin/bash
# launch_3_1.sh - 启动 3.1 仿真场景
# 机械臂在原点(0,0,0), 门框在(1,1,0)
set -e

IMAGE="${IMAGE:-isaac-sim-torch:6.0.1}"
CONTAINER="isaac_clean_3_1"
WORKSPACE="/home/dgx/air/test3.1"
SCRIPT="${SCRIPT:-clean_scene_3_1.py}"

echo "========================================================="
echo "[3.1] 仿真场景 (机械臂在原点, 门框在1,1,0)"
echo "[3.1] Image:    $IMAGE"
echo "[3.1] Script:   $SCRIPT"
echo "[3.1] Display:  :0"
echo "========================================================="

docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true

export DISPLAY=:0
xhost +local:docker 2>/dev/null || true

EXTRA_ARGS="${@:-}"

# 限制 CPU 线程数
KIT_ARGS="--/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4"

docker run -d \
  --name $CONTAINER \
  --entrypoint "/isaac-sim/python.sh" \
  --runtime=nvidia \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --ipc=host \
  --network=host \
  -e "ACCEPT_EULA=Y" \
  -e "DISPLAY=:0" \
  -e "XAUTHORITY=/root/.Xauthority" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$WORKSPACE:/workspace" \
  -w /workspace \
  $IMAGE \
  /workspace/${SCRIPT} ${EXTRA_ARGS} ${KIT_ARGS}

echo ""
echo "[3.1] 容器已启动"
echo "[3.1] 查看日志: docker logs -f $CONTAINER"
echo "[3.1] 停止: docker stop $CONTAINER"
echo "[3.1] 运行 conda: conda run -n 3.1 python /workspace/${SCRIPT}"
