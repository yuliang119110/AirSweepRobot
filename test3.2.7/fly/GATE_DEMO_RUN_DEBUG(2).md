# 无人机机械臂穿越门框：运行与调试脚本说明

## 目标场景

你当前搭建的场景按这个坐标理解：

```text
无人机 + 机械臂组合体水平起点: (0, 0)
门框水平中心:                 (-1, 1)
单位:                         米
实际飞行高度:                 z = 1.05 m
```

所以脚本中默认使用：

```bash
GATE_X=-1.0
GATE_Y=1.0
FLIGHT_Z=1.05
```

注意：你说的 `-110` 可以理解成水平坐标 `x=-1, y=1, z=0`。在 Isaac Sim 中如果门框中心真的放在 `z=0`，门框会贴地，所以演示脚本把“水平位置”和“飞行高度”分开，门框中心实际为 `(-1, 1, 1.05)`。

## 运行脚本

新增脚本：

```text
run_gate_demo.sh
```

在 Spark conda 环境中运行 GUI 演示：

```bash
cd /home/dgx/isaac_sim_test
chmod +x run_gate_demo.sh debug_gate_demo.sh

DRONE_ARM_URDF_PATH=/home/dgx/blander/drone_with_arm.urdf \
./run_gate_demo.sh gui
```

无 GUI 跑一次验证：

```bash
DRONE_ARM_URDF_PATH=/home/dgx/blander/drone_with_arm.urdf \
./run_gate_demo.sh headless
```

成功时日志会出现：

```text
[gate-demo] PASS: scripted gate fly-through completed
```

## 调试脚本

新增脚本：

```text
debug_gate_demo.sh
```

只做检查，不启动 Isaac Sim：

```bash
cd /home/dgx/isaac_sim_test

DRONE_ARM_URDF_PATH=/home/dgx/blander/drone_with_arm.urdf \
./debug_gate_demo.sh check
```

它会检查：

- `scripted_gate_flythrough.py` 是否存在
- `drone_with_arm.urdf` 是否存在
- `meshes/` 是否存在
- URDF 中有没有 `<collision>`
- 关键 mesh 是否存在
- Python 语法是否通过
- bash 脚本语法是否通过

做完整 headless 冒烟测试：

```bash
DRONE_ARM_URDF_PATH=/home/dgx/blander/drone_with_arm.urdf \
./debug_gate_demo.sh smoke
```

## 常用参数

临时调整门框位置：

```bash
GATE_X=-1.0 GATE_Y=1.0 ./run_gate_demo.sh gui
```

临时调整飞行高度：

```bash
FLIGHT_Z=1.2 ./run_gate_demo.sh gui
```

门框太小就加大：

```bash
GATE_WIDTH=1.6 GATE_HEIGHT=1.3 ./run_gate_demo.sh gui
```

模型朝向不对就加 yaw 偏移：

```bash
YAW_OFFSET_DEG=90 ./run_gate_demo.sh gui
```

或者：

```bash
YAW_OFFSET_DEG=-90 ./run_gate_demo.sh gui
```

穿过门框后飞远一点：

```bash
PASS_DISTANCE=1.8 ./run_gate_demo.sh gui
```

## 当前实现性质

这个版本是规则演示，不是强化学习，也不是真实物理飞控。

脚本会直接设置组合体根节点位姿：

```text
translate_op.Set(...)
rotate_op.Set(...)
```

因此它适合快速展示“无人机 + 机械臂合体穿过左前方门框”的效果。后续如果要做真实 RL，需要改成对无人机施加推力和力矩，让 PhysX 真实积分，并让碰撞体真实参与运动。
