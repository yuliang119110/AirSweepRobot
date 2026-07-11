#!/bin/bash
# Launch Isaac Sim 6.0.1 with composed scene
set -e

IMAGE="isaac-clean-rl:6.0.1"
CONTAINER="isaac_blander_view"
WORKDIR="/home/dgx/air/isaac_sim_test/blander"
SCENE="/workspace/isaac_sim_input/composed_scene.usd"

echo "========================================================"
echo " Launching Isaac Sim 6.0.1 with scene"
echo " Scene: $SCENE"
echo " Image: $IMAGE"
echo "========================================================"

# Clean up old container
docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true

export DISPLAY=:0
xhost +local:docker 2>/dev/null || true

# Run with direct scene loading
docker run -d \
  --name $CONTAINER \
  --gpus all \
  --ipc=host \
  --network=host \
  -e "ACCEPT_EULA=Y" \
  -e "DISPLAY=:0" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $WORKDIR:/workspace \
  -w /workspace \
  $IMAGE \
  /isaac-sim/python.sh -m isaacsim.app --no-window "$SCENE"

echo ""
echo "Container started. View logs: docker logs -f $CONTAINER"
echo "Stop: docker stop $CONTAINER"
