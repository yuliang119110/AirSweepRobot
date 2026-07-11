# Isaac Sim 仿真测试

> 本目录在 DGX 工作站（aarch64, GB10/Tegra iGPU）上运行 NVIDIA Isaac Sim，
> 验证 3D 物理场景渲染和 SO-ARM101 机械臂清洁场景的 RL 训练。

---

## 1. 环境准备

**硬件 / 系统**

| 项目 | 要求 |
|------|------|
| 硬件 | NVIDIA DGX（aarch64, GB10/Tegra iGPU） |
| 系统 | Linux，已安装 NVIDIA 驱动 |
| Docker | 支持 `--runtime=nvidia`（NVIDIA Container Toolkit） |
| 显示 | 本地物理显示器接在 `:0`，Isaac Sim 窗口直接渲染到屏幕 |

**一次性环境检查**

```bash
docker info | grep -i runtime       # 应看到 nvidia
nvidia-smi                          # GPU 可见
echo $DISPLAY                       # 应为 :0
```

**X11 授权**（每次重启 X 后执行一次）

```bash
xhost +local:docker
```

---

## 2. 拉取镜像

```bash
docker pull nvcr.io/nvidia/isaac-sim:6.0.1    # 推荐，约 17.5 GB
# 如需对比旧版本：
docker pull nvcr.io/nvidia/isaac-sim:5.1.0    # 约 19.6 GB
```

清洁场景 RL 和查看器需要自定义镜像（基于 6.0.1 构建）：

```bash
docker build -t isaac-clean-rl:6.0.1 -f isaac_sim_test/Dockerfile.clean_rl .
```

---

## 3. 启动（按用途选择）

### 3.1 物理场景可视化（入门，推荐先跑这个）

纯 USD + UsdPhysics 构建的物理场景：立方体塔、球体、圆柱、圆锥、地面、光照。
物理引擎会启动，物体在重力下运动。

```bash
# Isaac Sim 6.0.1（推荐）
bash isaac_sim_test/launch_601.sh

# Isaac Sim 5.1.0（旧版本对比）
bash isaac_sim_test/launch_510.sh
```

启动后看屏幕，约 25-30 秒后会出现 Isaac Sim 窗口，可看到 3D 物体。

确认启动成功的日志标志：
```
[xx.xxxs] Simulation App Startup Complete
[xx.xxxs] app ready
```

### 3.2 SO-ARM101 机械臂清洁场景 RL

```bash
# 随机策略可视化（首次验证推荐）
bash isaac_sim_test/launch_clean_rl.sh random

# 平滑正弦运动可视化（低 CPU 验证）
bash isaac_sim_test/launch_clean_rl.sh motion

# 推理（需要已训练的 checkpoint）
bash isaac_sim_test/launch_clean_rl.sh play

# RL 训练（headless，无窗口）
bash isaac_sim_test/launch_clean_rl.sh train
```

训练时可调环境数：
```bash
NUM_ENVS=4 bash isaac_sim_test/launch_clean_rl.sh train
```

### 3.3 机械臂清洁场景查看器（低 CPU）

纯 SimulationApp + USD API，加载机械臂模型并动画关节，CPU 占用更低：

```bash
# 默认运行 clean_scene.py
bash isaac_sim_test/launch_clean_viewer.sh

# 指定其他脚本
SCRIPT=clean_scene_viewer.py bash isaac_sim_test/launch_clean_viewer.sh
```

---

## 4. 启动后的操作

**查看日志**
```bash
docker logs -f isaac_601_test         # 物理场景
docker logs -f isaac_clean_rl         # RL
docker logs -f isaac_clean_viewer     # 查看器
```

**截图确认画面**
```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  gnome-screenshot -f /tmp/screen.png
```

**停止容器**
```bash
docker stop isaac_601_test isaac_clean_rl isaac_clean_viewer
```

**进入运行中的容器**
```bash
docker exec -it isaac_601_test bash
```

---

## 5. 常见问题

### 视口灰色 / 黑屏（已修复）

**现象**：Isaac Sim 窗口打开但视口一片灰色或黑色，看不到 3D 物体。

**根因**：脚本里用了 `omni.kit.commands.execute("Play")`，该命令在 5.1.0 和 6.0.1
中均未注册，物理引擎不启动，视口保持灰色。**不是镜像版本问题。**

**修复**（已在 [sim_test.py](sim_test.py) 中应用）：

```python
import omni.timeline

timeline = omni.timeline.get_timeline_interface()
timeline.play()
```

### Docker 报错：unknown flag --runtime

未安装或未启用 NVIDIA Container Toolkit。安装后重启 Docker：
```bash
sudo systemctl restart docker
```

### 窗口不显示 / Cannot connect to display

确认 `$DISPLAY` 为 `:0`，并已执行 `xhost +local:docker`。

### 启动很慢

首次启动需要 25-30 秒加载扩展，属正常。如果超过 60 秒，检查 `docker logs` 是否有报错。

### GPU 显存不足

清洁场景查看器已限制线程数为 4。如仍不足，减少 RL 环境数：
```bash
NUM_ENVS=1 bash isaac_sim_test/launch_clean_rl.sh train
```

---

## 6. 文件说明

| 文件 | 说明 |
|------|------|
| [sim_test.py](sim_test.py) | 物理场景主脚本（纯 USD + UsdPhysics API） |
| [launch_601.sh](launch_601.sh) | 物理场景启动 — Isaac Sim 6.0.1 |
| [launch_510.sh](launch_510.sh) | 物理场景启动 — Isaac Sim 5.1.0 |
| [clean_scene.py](clean_scene.py) | SO-ARM101 清洁场景脚本（加载机械臂 + 关节动画） |
| [clean_scene_viewer.py](clean_scene_viewer.py) | 清洁场景可视化查看器 |
| [launch_clean_rl.sh](launch_clean_rl.sh) | 清洁场景 RL 启动（random / motion / play / train） |
| [launch_clean_viewer.sh](launch_clean_viewer.sh) | 清洁场景低 CPU 查看器启动 |
| [launch_gui.sh](launch_gui.sh) | 直接启动 GUI（自定义镜像，旧脚本） |
| [Dockerfile.clean_rl](Dockerfile.clean_rl) | 清洁场景 RL 镜像构建文件 |

### sim_test.py 场景内容

| 元素 | 说明 |
|------|------|
| 物理场景 | 重力 9.81 m/s^2，方向 -Z |
| 地面 | 带碰撞的 `UsdGeom.Plane`，灰色 |
| 立方体塔 | 5 个橙色立方体（0.1m），刚体 + 碰撞 |
| 球体 | 6 个蓝色球体（半径 0.06m），圆周分布 |
| 圆柱 | 4 个紫色圆柱，刚体 + 碰撞 |
| 圆锥 | 1 个黄色圆锥，装饰无物理 |
| 光照 | 平行光 800 + 穹顶光 400 |
