# Isaac Sim 仿真测试

在 DGX 工作站（aarch64 / GB10 Tegra iGPU）上运行 NVIDIA Isaac Sim，验证 3D 物理场景渲染与交互。

## 环境要求

| 项目 | 要求 |
|------|------|
| 硬件 | NVIDIA DGX（aarch64, GB10/Tegra iGPU） |
| 操作系统 | Linux（已安装 NVIDIA 驱动） |
| Docker | 支持 `--runtime=nvidia`（NVIDIA Container Toolkit） |
| X11 | 本地显示 `:0`，`xhost +local:docker` 授权 |

## Docker 镜像

本地已缓存以下两个版本，均可使用：

- `nvcr.io/nvidia/isaac-sim:6.0.1`（17.5 GB）— 推荐使用
- `nvcr.io/nvidia/isaac-sim:5.1.0`（19.6 GB）— 旧版本

两个镜像均通过 `--entrypoint ""` + `/isaac-sim/python.sh` 方式直接运行 Python 脚本。

## 快速开始

```bash
# 启动 6.0.1 场景（推荐）
bash isaac_sim_test/launch_601.sh

# 或启动 5.1.0 场景
bash isaac_sim_test/launch_510.sh

# 查看日志
docker logs -f isaac_601_test
```

容器以 detached 模式运行。日志出现以下两行即表示启动成功：

```
[xx.xxxs] Simulation App Startup Complete
[xx.xxxs] app ready
```

渲染画面会直接输出到本地显示器（`DISPLAY=:0`）。

## 文件说明

| 文件 | 用途 |
|------|------|
| [sim_test.py](sim_test.py) | 主测试脚本：纯 USD + UsdPhysics API 创建物理场景（立方体塔、球体、圆柱、圆锥、地面、光照） |
| [launch_601.sh](launch_601.sh) | Isaac Sim 6.0.1 Docker 启动脚本 |
| [launch_510.sh](launch_510.sh) | Isaac Sim 5.1.0 Docker 启动脚本 |
| [launch_gui.sh](launch_gui.sh) | 直接启动 Isaac Sim GUI（无自定义脚本） |
| [clean_scene.py](clean_scene.py) | SO-ARM101 机械臂清洁场景 RL 脚本 |
| [clean_scene_viewer.py](clean_scene_viewer.py) | 清洁场景可视化查看器 |
| [launch_clean_rl.sh](launch_clean_rl.sh) | 清洁场景 RL 训练启动脚本 |
| [launch_clean_viewer.sh](launch_clean_viewer.sh) | 清洁场景查看器启动脚本 |
| [Dockerfile.clean_rl](Dockerfile.clean_rl) | 清洁场景 RL 自定义镜像构建文件 |

## sim_test.py 场景内容

脚本通过 `SimulationApp({"headless": False})` 启动 GUI 模式，使用纯 USD API 构建：

- **物理场景**：重力 9.81 m/s^2，方向 -Z
- **地面平面**：带碰撞的 `UsdGeom.Plane`，灰色
- **堆叠立方体塔**：5 个橙色立方体（0.1m），带刚体+碰撞
- **散布球体**：6 个蓝色球体（半径 0.06m），圆周分布
- **散布圆柱**：4 个紫色圆柱，带刚体+碰撞
- **装饰圆锥**：1 个黄色圆锥，无物理
- **光照**：平行光（800 强度）+ 穹顶光（400 强度）

## 已知问题与修复记录

### 视口灰色/黑屏问题（已修复）

**现象**：Isaac Sim 视口渲染为灰色/黑色，看不到任何 3D 物体。

**排查过程**：

1. 先后测试了 6.0.1 和 5.1.0 两个镜像版本 — 均出现相同问题，排除镜像版本因素。
2. 检查 Docker 容器启动日志，发现脚本中 `omni.kit.commands.execute("Play")` 报未注册命令错误。
3. 该命令在两个版本中均未注册，导致物理引擎无法启动，视口保持灰色。

**根因**：脚本 API 错误，不是容器版本问题。

**修复**：在 [sim_test.py](sim_test.py) 中：

1. 导入部分添加 `import omni.timeline`
2. 将 `omni.kit.commands.execute("Play")` 替换为：

```python
timeline = omni.timeline.get_timeline_interface()
timeline.play()
```

修复后两个版本的镜像均可正常渲染。

## 截图验证

```bash
# 在本地截图查看当前画面
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  gnome-screenshot -f /tmp/screen_check.png
```

## 容器管理

```bash
# 停止容器
docker stop isaac_601_test

# 进入运行中的容器
docker exec -it isaac_601_test bash

# 查看实时日志
docker logs -f isaac_601_test
```
