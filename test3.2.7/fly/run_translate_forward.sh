#!/bin/bash
# 启动无人机向前平移 N 米演示
set -e

IMAGE="${IMAGE:-isaac-sim-torch:6.0.1}"
CONTAINER="isaac_translate_forward"
SCRIPT="${SCRIPT:-translate_forward.py}"
WORKSPACE="/home/dgx/air/test3.2"

# 平移参数（可通过环境变量覆盖）
START_X="${START_X:-0.0}"
START_Y="${START_Y:-0.0}"
START_Z="${START_Z:-1.05}"
YAW_DEG="${YAW_DEG:-0.0}"
FORWARD_METERS="${FORWARD_METERS:-3.0}"
YAW_OFFSET_DEG="${YAW_OFFSET_DEG:-0.0}"
MAX_FRAMES="${MAX_FRAMES:-300}"
LOOP="${LOOP:-}"
HEADLESS="${HEADLESS:-}"

echo "=========================================================="
echo "[translate] 无人机向前平移演示"
echo "[translate] Image:      $IMAGE"
echo "[translate] Script:     $SCRIPT"
echo "[translate] Start:      ($START_X, $START_Y, $START_Z)"
echo "[translate] Yaw:        ${YAW_DEG} deg"
echo "[translate] Forward:    ${FORWARD_METERS} m"
echo "=========================================================="

# 清理旧容器
docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true

export DISPLAY=:0
xhost +local:docker 2>/dev/null || true

SCRIPT_ARGS=""
[[ -n "$LOOP" ]] && SCRIPT_ARGS="$SCRIPT_ARGS --loop"
[[ -n "$HEADLESS" ]] && SCRIPT_ARGS="$SCRIPT_ARGS --headless"

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
  /workspace/fly/${SCRIPT} \
  --start-x ${START_X} --start-y ${START_Y} --start-z ${START_Z} \
  --yaw-deg ${YAW_DEG} \
  --forward-meters ${FORWARD_METERS} \
  --yaw-offset-deg ${YAW_OFFSET_DEG} \
  --max-frames ${MAX_FRAMES} \
  ${SCRIPT_ARGS} ${KIT_ARGS}

echo ""
echo "[translate] 容器已启动，Isaac Sim 窗口应显示在物理显示器 :0 上"
echo "[translate] 查看日志: docker logs -f $CONTAINER"
echo "[translate] 停止: docker stop $CONTAINER"
echo ""
echo "[translate] 提示：可通过环境变量自定义参数，例如："
echo "  YAW_DEG=45 FORWARD_METERS=5 bash fly/run_translate_forward.sh"
