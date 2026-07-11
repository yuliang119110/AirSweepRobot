# 🚁 Tilted-Octorotor: Open-Source Tilted-Rotor Octocopter Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-orange.svg)](https://github.com/google/jax)
[![PX4](https://img.shields.io/badge/PX4-Compatible-green.svg)](https://px4.io/)
[![Stars](https://img.shields.io/github/stars/yourusername/tilted-octorotor?style=social)](https://github.com/yourusername/tilted-octorotor/stargazers)

> **A novel tilted-rotor octocopter platform with 30° rotor inclination, featuring Sample-based MPC control and full hardware/software open-source design for embodied AI and aerial manipulation research.**

[English](#english) | [中文](#中文)

---

## 🌟 Highlights

- 🔧 **Innovative Design**: 8 rotors with 30° tilt angle for enhanced maneuverability
- 🤖 **Sample-based MPC**: JAX-accelerated parallel optimization with 2048 samples
- 📐 **Complete CAD Files**: SolidWorks models (`.sldasm`) for easy replication
- 🛠️ **Hardware BOM**: Detailed component list with purchase links
- 📊 **Real-time Visualization**: 6 comprehensive PNG plots for analysis
- 🧪 **Disturbance Testing**: 6-axis force/torque sensor integration
- 🚀 **Embodied AI Ready**: Designed for "Flight + Embodiment + Household" scenarios

---

## 📸 Gallery

<div align="center">
  <img src="docs/images/octorotor_cad.png" width="45%" alt="CAD Model"/>
  <img src="docs/images/trajectory_3d.png" width="45%" alt="3D Trajectory"/>
</div>

<div align="center">
  <img src="docs/images/thrust_allocation.png" width="45%" alt="Thrust Allocation"/>
  <img src="docs/images/xy_trajectory.png" width="45%" alt="XY Trajectory"/>
</div>

---

## 🎯 Why Tilted Rotors?

Traditional multi-rotors generate thrust only in the vertical direction, limiting their ability to:
- ❌ Resist horizontal disturbances without tilting
- ❌ Perform aggressive maneuvers efficiently
- ❌ Manipulate objects while maintaining stability

**Our tilted-rotor design (30° inclination) enables:**
- ✅ **Direct horizontal force generation** without body tilt
- ✅ **Enhanced disturbance rejection** (tested with ±5N external forces)
- ✅ **Improved manipulation capability** for embodied AI tasks
- ✅ **Omnidirectional thrust vectoring** for precise control

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Control Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│  State Estimation  →  Sample-based MPC  →  Mixer  →  ESCs  │
│       (IMU)             (JAX/GPU)         (6×8)    (8×ESC)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Hardware Components                        │
├─────────────────────────────────────────────────────────────┤
│  • Flight Controller: USX51 (Quad-core ARM + GPU)          │
│  • Motors: EMAX ECO II 2207 1700KV (×8)                    │
│  • Props: HQProp 5043 V2S Tri-blade (×8)                   │
│  • ESCs: LANRC 35A BLHeli_32 (×8)                          │
│  • Force Sensor: HKVTech 6-axis F/T sensor                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/tilted-octorotor.git
cd tilted-octorotor
```

### 2. Install Dependencies

```bash
pip install jax jaxlib numpy matplotlib
```

### 3. Run Simulation

```bash
python3 tilted_octorotor_mpc_simple.py
```

**Output**: 6 PNG plots showing position, velocity, thrust allocation, disturbances, and trajectories.

### 4. Build Hardware (Optional)

See [Hardware Guide](docs/HARDWARE.md) for:
- 📐 CAD files and 3D printing instructions
- 🛒 Component purchase links
- 🔧 Assembly tutorial
- ⚡ Wiring diagram

---

## 📊 Simulation Results

The Sample-based MPC controller successfully maintains hover under ±5N random disturbances:

| Metric | Value |
|--------|-------|
| **Mean Position Error** | 0.8-1.2 m |
| **Control Frequency** | 20 Hz |
| **MPC Samples** | 512-2048 |
| **Computation Time** | 8-10 ms/step (after JIT) |
| **Disturbance Rejection** | ±5N force, ±0.1N·m torque |

---

## 🔬 Technical Details

### Rotor Configuration (Based on PX4 8001_octo_x)

```
        2       7
         \     /
          \   /
    3 ---- + ---- 0
          /   \
         /     \
        5       4
           1
           6
```

**Tilt Angle**: 30° (relative to horizontal plane)  
**Thrust Direction**: Tilted towards rotor position relative to center

### Mixer Matrix

The 6×8 mixer matrix maps rotor thrusts to body wrench:

```
[Fx, Fy, Fz, Mx, My, Mz]ᵀ = M × [T₀, T₁, ..., T₇]ᵀ
```

Where:
- `Fx, Fy, Fz`: Total force in body frame
- `Mx, My, Mz`: Total torque
- `T₀, ..., T₇`: Individual rotor thrusts

### Sample-based MPC Algorithm

1. **Sample** N thrust sequences around hover point
2. **Rollout** N trajectories in parallel (JAX vmap)
3. **Evaluate** cost function for each trajectory
4. **Select** minimum-cost trajectory
5. **Execute** first control input
6. **Repeat** with warm start

**Cost Function**:
```
J = Σ(Q_pos·||p - p*||² + Q_vel·||v - v*||² + R·||u - u_hover||²)
```

---

## 🛠️ Hardware Specifications

### Bill of Materials (BOM)

| Component | Model | Quantity | Unit Price | Link |
|-----------|-------|----------|------------|------|
| **Flight Controller** | USX51 Computing Power FC | 1 | ~$XXX | [MakerFire](https://shop.makerfire.com/en-jp/pages/usx51-computing-power-flight-controller) |
| **Motors** | EMAX ECO II 2207 1700KV | 8 | ~$15 | [Link](#) |
| **Propellers** | HQProp 5043 V2S Tri-blade | 8 | ~$2 | [Link](#) |
| **ESCs** | LANRC 35A BLHeli_32 | 8 | ~$12 | [Link](#) |
| **Force Sensor** | HKVTech 6-axis F/T | 1 | ~$XXX | [HKVTech](https://www.hkvtech.cn/) |
| **Frame** | Custom 3D Printed | 1 | ~$20 | See CAD files |
| **Battery** | 4S LiPo 5000mAh | 1 | ~$40 | [Link](#) |

**Total Cost**: ~$XXX USD (excluding tools)

### Key Specifications

| Parameter | Value |
|-----------|-------|
| **Total Weight** | ~1.0 kg |
| **Max Thrust** | ~78.5 N (8 × 9.81N) |
| **Thrust-to-Weight** | ~8:1 |
| **Flight Time** | ~8-12 min (estimated) |
| **Rotor Diameter** | 5 inch (127 mm) |
| **Frame Size** | ~400 mm (diagonal) |

---

## 📁 Repository Structure

```
tilted-octorotor/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── docs/
│   ├── HARDWARE.md                    # Hardware build guide
│   ├── SOFTWARE.md                    # Software setup guide
│   ├── THEORY.md                      # Theoretical background
│   └── images/                        # Documentation images
├── cad/
│   ├── octorotor_frame.sldasm        # SolidWorks assembly
│   ├── rotor_mount.sldprt            # Rotor mount part
│   └── stl/                          # STL files for 3D printing
├── software/
│   ├── tilted_octorotor_mpc_simple.py  # Main simulation
│   ├── mixer.py                       # Mixer implementation
│   ├── mpc_controller.py              # MPC controller
│   └── dynamics.py                    # Dynamics model
├── firmware/
│   ├── px4_config/                    # PX4 configuration files
│   └── usx51_setup/                   # USX51 setup scripts
├── hardware/
│   ├── bom.csv                        # Bill of materials
│   ├── wiring_diagram.pdf             # Wiring diagram
│   └── assembly_guide.pdf             # Assembly instructions
└── results/
    ├── 01_position_velocity.png       # Simulation results
    ├── 02_trajectory_3d.png
    └── ...
```

---

## 🎓 Research Applications

This platform is designed for cutting-edge research in:

### 1. Embodied AI
- 🏠 **Household Robotics**: Object manipulation in domestic environments
- 🤝 **Human-Robot Interaction**: Safe physical interaction
- 🎯 **Task Planning**: High-level reasoning for complex tasks

### 2. Aerial Manipulation
- 🔧 **Contact-based Manipulation**: Push, pull, grasp objects
- 🎨 **Painting/Cleaning**: Surface interaction tasks
- 📦 **Package Delivery**: Precise placement and retrieval

### 3. Advanced Control
- 🧮 **Learning-based Control**: RL/IL for complex behaviors
- 🎯 **Optimal Control**: Trajectory optimization
- 🛡️ **Robust Control**: Disturbance rejection

---

## 🤝 Acknowledgments

We would like to express our sincere gratitude to:

- **[USX51 Flight Controller](https://shop.makerfire.com/en-jp/pages/usx51-computing-power-flight-controller)** by MakerFire - for providing the powerful computing platform with quad-core ARM + GPU
- **[PX4 Autopilot Project](https://px4.io/)** - for the excellent open-source flight stack and airframe configurations
- **[HKVTech (航凯微电)](https://www.hkvtech.cn/)** - for providing the 6-axis force/torque sensor for disturbance testing
- **[BAAI Maker Marathon](https://hub.baai.ac.cn/view/48654)** - for supporting our exploration of "Flight + Embodiment + Household" scenarios

This project is part of our ongoing research in embodied AI and aerial robotics. We welcome contributions and collaborations!

---

## 📖 Citation

If you use this platform in your research, please cite:

```bibtex
@misc{tilted_octorotor_2025,
  title={Tilted-Octorotor: An Open-Source Tilted-Rotor Platform for Embodied AI},
  author={Your Name and Team},
  year={2025},
  publisher={GitHub},
  howpublished={\\url{https://github.com/yourusername/tilted-octorotor}}
}
```

---

## 🌐 Related Projects

- **[FluxTide](https://github.com/DataFlux-Robot/FluxTide)** - Sample-based MPC for humanoid robots
- **[DIAL-MPC](https://github.com/LeCAR-Lab/dial-mpc)** - Diffusion-inspired annealing for legged MPC
- **[PX4 Autopilot](https://github.com/PX4/PX4-Autopilot)** - Open-source flight control software
- **[Brax](https://github.com/google/brax)** - Differentiable physics engine

---

## 🛣️ Roadmap

- [x] Release simulation code
- [x] Publish CAD files
- [x] Document hardware BOM
- [ ] Hardware prototype testing
- [ ] PX4 firmware integration
- [ ] Real-world flight experiments
- [ ] Learning-based control (RL/IL)
- [ ] Object manipulation demos
- [ ] ROS 2 integration
- [ ] Multi-agent coordination

---

## 🤝 Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star this repository!

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **Project Lead**: [Your Name](mailto:your.email@example.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/tilted-octorotor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/tilted-octorotor/discussions)

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/tilted-octorotor&type=Date)](https://star-history.com/#yourusername/tilted-octorotor&Date)

---

<div align="center">
  
### 🚀 If you find this project useful, please consider giving it a ⭐!

**Made with ❤️ for the robotics community**

[⬆ Back to Top](#-tilted-octorotor-open-source-tilted-rotor-octocopter-platform)

</div>

---

# 中文

## 🚁 倾斜旋翼八旋翼:开源倾斜旋翼多旋翼平台

> **一个创新的30°倾斜旋翼八旋翼平台,配备Sample-based MPC控制器和完整的硬件/软件开源设计,专为具身智能和空中操作研究而设计。**

## 🌟 项目亮点

- 🔧 **创新设计**: 8个旋翼30°倾斜,增强机动性
- 🤖 **Sample-based MPC**: JAX加速并行优化,支持2048样本
- 📐 **完整CAD文件**: SolidWorks模型(`.sldasm`),易于复制
- 🛠️ **硬件清单**: 详细的元器件列表和购买链接
- 📊 **实时可视化**: 6张综合PNG图表用于分析
- 🧪 **扰动测试**: 集成六分量力/力矩传感器
- 🚀 **具身AI就绪**: 专为"飞行+具身+家务场景"设计

## 🎯 为什么选择倾斜旋翼?

传统多旋翼只能产生垂直推力,限制了其能力:
- ❌ 无法在不倾斜的情况下抵抗水平扰动
- ❌ 无法高效执行激进机动
- ❌ 无法在保持稳定的同时操纵物体

**我们的倾斜旋翼设计(30°倾角)实现了:**
- ✅ **直接产生水平力**,无需机体倾斜
- ✅ **增强扰动抑制**(测试±5N外力)
- ✅ **改进操作能力**,适用于具身AI任务
- ✅ **全向推力矢量**,实现精确控制

## 🤝 致谢

特别感谢:

- **[USX51飞控](https://shop.makerfire.com/en-jp/pages/usx51-computing-power-flight-controller)** - 提供强大的四核ARM+GPU计算平台
- **[PX4项目](https://px4.io/)** - 优秀的开源飞控软件和机架配置
- **[航凯微电公司](https://www.hkvtech.cn/)** - 提供六分量力/力矩传感器
- **[创客松项目](https://hub.baai.ac.cn/view/48654)** - 支持我们探索"飞行+具身+家务场景"

本项目是我们在具身AI和空中机器人领域持续研究的一部分,欢迎贡献和合作!

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/tilted-octorotor.git
cd tilted-octorotor
```

### 2. 安装依赖
```bash
pip install jax jaxlib numpy matplotlib
```

### 3. 运行仿真
```bash
python3 tilted_octorotor_mpc_simple.py
```

**输出**: 6张PNG图表,显示位置、速度、推力分配、扰动和轨迹。

## 🛠️ 硬件规格

### 物料清单(BOM)

| 组件 | 型号 | 数量 | 单价 |
|------|------|------|------|
| **飞控** | USX51算力飞控 | 1 | ~¥XXX |
| **电机** | EMAX ECO II 2207 1700KV | 8 | ~¥100 |
| **螺旋桨** | HQProp 5043 V2S三叶桨 | 8 | ~¥15 |
| **电调** | LANRC 35A BLHeli_32 | 8 | ~¥80 |
| **力传感器** | 航凯微电六分量天平 | 1 | ~¥XXX |
| **机架** | 定制3D打印 | 1 | ~¥150 |

**总成本**: 约¥XXX元(不含工具)

## 📄 开源协议

本项目采用MIT协议 - 详见[LICENSE](LICENSE)文件。

---

<div align="center">

### 🚀 如果您觉得这个项目有用,请给我们一个⭐!

**为机器人社区用❤️制作**

</div>

