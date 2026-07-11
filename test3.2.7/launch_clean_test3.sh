 #!/bin/bash
 # launch_clean_test3.sh - 启动 test3 清洁仿真场景
 # 导入 drone_with_arm URDF 和 base_basic_pbr(2).usdz
 set -e
 
 IMAGE="${IMAGE:-isaac-sim-torch:6.0.1}"
 CONTAINER="isaac_clean_test3"
 WORKSPACE="/home/dgx/air/test3"
 SCRIPT="${SCRIPT:-clean_scene_test3.py}"
 
 echo "========================================================="
 echo "[test3] 清洁仿真场景 (drone_with_arm + base_basic_pbr)"
 echo "[test3] Image:    $IMAGE"
 echo "[test3] Script:   $SCRIPT"
 echo "[test3] Display:  :0"
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
 echo "[test3] 容器已启动,Isaac Sim 窗口应显示在物理显示器 :0 上"
 echo "[test3] 查看日志: docker logs -f $CONTAINER"
 echo "[test3] 停止: docker stop $CONTAINER"
