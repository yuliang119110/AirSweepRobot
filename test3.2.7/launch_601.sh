#!/bin/bash
set -e
CONTAINER="isaac_601_test"
WORKSPACE="/home/dgx/air/isaac_sim_test"

docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true

export DISPLAY=:0
xhost +local:docker 2>/dev/null || true

docker run -d \
  --name $CONTAINER \
  --entrypoint "" \
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
  nvcr.io/nvidia/isaac-sim:6.0.1 \
  /isaac-sim/python.sh /workspace/sim_test.py

echo "6.0.1 container started"
echo "Logs: docker logs -f $CONTAINER"
