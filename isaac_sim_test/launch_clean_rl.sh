#!/bin/bash
# launch_clean_rl.sh - 在 Isaac Sim 6.0.1 + Isaac Lab 3.0 中运行 SO-ARM101 清洁场景
#
# 用法:
#   ./launch_clean_rl.sh random  # 随机策略可视化（推荐首次验证，GUI，2 环境）
#   ./launch_clean_rl.sh motion  # 平滑正弦运动可视化（低CPU验证）
#   ./launch_clean_rl.sh play    # 推理可视化（需要已训练的 checkpoint）
#   ./launch_clean_rl.sh train   # RL 训练（headless）
set -e

IMAGE="isaac-clean-rl:6.0.1"
CONTAINER="isaac_clean_rl"
MODEL_TEST="/home/dgx/model_test"

MODE="${1:-random}"
NUM_ENVS="${NUM_ENVS:-1}"

echo "========================================================="
echo "[RL] SO-ARM101 清洁场景 RL ($MODE)"
echo "[RL] Image: $IMAGE (Isaac Sim 6.0.1 + Isaac Lab 3.0)"
echo "[RL] Num envs: $NUM_ENVS"
echo "========================================================="

docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true

export DISPLAY=:0
xhost +local:docker 2>/dev/null || true

if [ "$MODE" = "train" ]; then
   TASK="Isaac-SO-ARM101-Clean-v0"
   EXTRA="--headless --viz none"
else
   TASK="Isaac-SO-ARM101-Clean-Play-v0"
   EXTRA="--viz kit"
fi

if [ "$MODE" = "random" ]; then
    SCRIPT="isaac_so_arm101.scripts.random_agent"
elif [ "$MODE" = "motion" ]; then
    SCRIPT="isaac_so_arm101.scripts.gentle_motion"
else
    SCRIPT="isaac_so_arm101.scripts.rsl_rl.$MODE"
fi

docker run -d \
  --name $CONTAINER \
  --entrypoint /bin/bash \
  --gpus all \
  --ipc=host \
  --network=host \
  -e "ACCEPT_EULA=Y" \
  -e "DISPLAY=:0" \
  -e "XAUTHORITY=/root/.Xauthority" \
  -e "SO_ARM101_CLEAN_ASSET_DIR=/model_test/机械臂" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$MODEL_TEST:/model_test" \
  $IMAGE \
  -c '
    set -e
    echo "[RL] Installing isaac_so_arm101 extension..."
    /isaac-sim/python.sh -m pip install -e /model_test/isaac_so_arm101 --no-deps -q 2>&1 | tail -2
    EXTRA_ARGS="'"$EXTRA"'"
    if [ "'"$MODE"'" = "play" ]; then
      EXTRA_ARGS="'"$EXTRA"' --real-time"
    fi
    echo "[RL] Launching: '"$SCRIPT"' --task '"$TASK"' --num_envs '"$NUM_ENVS"' ${EXTRA_ARGS} --disable_fabric"
    cd /model_test/isaac_so_arm101/src
    /isaac-sim/python.sh -m '"$SCRIPT"' --task '"$TASK"' --num_envs '"$NUM_ENVS"' ${EXTRA_ARGS}
  '

echo ""
echo "[RL] 容器已启动"
echo "[RL] 查看日志: docker logs -f $CONTAINER"
echo "[RL] 停止: docker stop $CONTAINER"
