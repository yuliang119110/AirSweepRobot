#!/bin/bash
# launch_translate.sh — 启动无人机+机械臂向前平移穿越门框演示
#
# 用法:
#   bash launch_translate.sh              # GUI 模式，跑一次（300帧）
#   LOOP=1   bash launch_translate.sh     # GUI 模式，循环跑
#   HEADLESS=1 bash launch_translate.sh   # 无画面模式
#
# 环境变量覆盖:
#   FORWARD_METERS  平移距离（默认 3.0 米）
#   YAW_DEG         起始偏航角（度，默认 0）
#   START_Z         飞行高度（默认 1.05 米）
#   MAX_FRAMES      总帧数（默认 300）
#   BASE_X/Y/Z      门框位置（默认 1.5, 0, 0）

set -e

IMAGE="${IMAGE:-isaac-sim-torch:6.0.1}"
CONTAINER="isaac_translate_forward"
SCRIPT="translate_forward.py"
WORKSPACE="/home/dgx/air/test3.2.7"

# ========== 场景参数 ==========
START_X="${START_X:-0.0}"
START_Y="${START_Y:-0.0}"
START_Z="${START_Z:-1.05}"
YAW_DEG="${YAW_DEG:-0.0}"
FORWARD_METERS="${FORWARD_METERS:-3.0}"
YAW_OFFSET_DEG="${YAW_OFFSET_DEG:-0.0}"
MAX_FRAMES="${MAX_FRAMES:-300}"
LOOP="${LOOP:-}"
HEADLESS="${HEADLESS:-}"

# ========== 模型路径 ==========
URDF_PATH="${URDF_PATH:-/workspace/blander/drone_with_arm/drone_with_arm.urdf}"
BASE_USDZ_PATH="${BASE_USDZ_PATH:-/workspace/blander/base_basic_pbr(2).usdz}"

# 门框位置
BASE_X="${BASE_X:-1.5}"
BASE_Y="${BASE_Y:-0.0}"
BASE_Z="${BASE_Z:-0.0}"

# ========== 打印配置 ==========
echo "=========================================================="
echo "  无人机+机械臂向前平移穿越门框"
echo "  工作区:    $WORKSPACE"
echo "  URDF:      $URDF_PATH"
echo "  门框:      $BASE_USDZ_PATH"
echo "  起点:      ($START_X, $START_Y, $START_Z)"
echo "  平移:      向前 ${FORWARD_METERS} 米"
echo "  偏航:      ${YAW_DEG}°"
echo "  门框位置:  ($BASE_X, $BASE_Y, $BASE_Z)"
echo "  循环:      ${LOOP:-否}"
echo "  Headless:  ${HEADLESS:-否}"
echo "=========================================================="

# ========== 清理旧容器 ==========
docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true

export DISPLAY=:0
xhost +local:docker 2>/dev/null || true

# ========== 构造参数 ==========
SCRIPT_ARGS=""
[[ -n "$LOOP" ]] && SCRIPT_ARGS="$SCRIPT_ARGS --loop"
[[ -n "$HEADLESS" ]] && SCRIPT_ARGS="$SCRIPT_ARGS --headless"

KIT_ARGS="--/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4"

# ========== 启动容器 ==========
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
  --urdf-path "${URDF_PATH}" \
  --base-usdz-path "${BASE_USDZ_PATH}" \
  --start-x ${START_X} --start-y ${START_Y} --start-z ${START_Z} \
  --yaw-deg ${YAW_DEG} \
  --forward-meters ${FORWARD_METERS} \
  --yaw-offset-deg ${YAW_OFFSET_DEG} \
  --base-x ${BASE_X} --base-y ${BASE_Y} --base-z ${BASE_Z} \
  --max-frames ${MAX_FRAMES} \
  ${SCRIPT_ARGS} ${KIT_ARGS}

# ========== 输出操作指引 ==========
echo ""
echo "  容器已启动（名称: $CONTAINER）"
echo ""
echo "  查看日志:  docker logs -f $CONTAINER"
echo "  停止:      docker stop $CONTAINER"
echo ""
echo "  自定义示例:"
echo "    FORWARD_METERS=5  HEADLESS=1  bash $0"
echo "    YAW_DEG=45  FORWARD_METERS=2  LOOP=1  bash $0"
