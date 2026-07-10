# LocateAnything-3B 本地部署 (NVIDIA DGX Spark / GB10)

基于 NVIDIA [LocateAnything](https://huggingface.co/spaces/nvidia/LocateAnything) 的本地化部署，
适配 NVIDIA DGX Spark (Grace Blackwell GB10, 128GB 统一内存, aarch64)。

## 快速开始

```bash
cd /home/dgx/locate-anything/LocateAnything
./run.sh
```

服务启动后访问 http://127.0.0.1:7860

首次启动会触发 torch.compile 编译 + warmup 预热，约需 60-90 秒。
编译结果会缓存，后续重启编译时间大幅缩短。

## 环境说明

| 项目 | 版本/路径 |
|------|-----------|
| Conda 环境 | `locate-anything` (Python 3.10) |
| PyTorch | 2.13.0+cu130 |
| Transformers | 4.57.1 |
| Gradio | 6.17.3 |
| GPU | NVIDIA GB10 (Blackwell, CC 12.1) |
| 模型缓存 | `/home/dgx/locate-anything/hf_cache/` (~7.3GB) |

## 本地适配文件

以下两个文件是为了在 aarch64 / 本地 GPU 环境运行而添加的，HF 原始 Space 不包含：

- `spaces.py` — 替代 HF ZeroGPU 的 `@spaces.GPU` 装饰器，本地直接 no-op
- `decord.py` — decord 无 aarch64 wheel，stub 掉视频读取模块（图像功能不受影响）

## 配置项 (环境变量)

所有优化开关通过环境变量控制，`run.sh` 已设置默认值。
可以在启动前覆盖任意变量：

```bash
# 关闭 torch.compile
LA_COMPILE=0 ./run.sh

# 开启 FP8 量化
LA_FP8=1 ./run.sh

# 只开 L0，最安全的调试模式
LA_COMPILE=0 LA_FP8=0 LA_WARMUP=0 ./run.sh
```

### 完整配置表

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MODEL_PATH` | `nvidia/LocateAnything-3B` | 模型名称或本地路径 |
| `LA_GEN_MODE` | `hybrid` | 生成模式: `hybrid` / `fast` / `slow` |
| `LA_TF32` | `1` | L0: 开启 TF32 矩阵乘 (单图 +28% tok/s) |
| `LA_CUDNN_BENCH` | `1` | L0: cuDNN benchmark 自动调优 |
| `LA_COMPILE` | `1` | L1: torch.compile 编译优化 |
| `LA_COMPILE_MODE` | `reduce-overhead` | L1: 编译模式 (`reduce-overhead` / `default`) |
| `LA_FP8` | `0` | L2: FP8 (float8_e4m3fn) 权重量化 |
| `LA_WARMUP` | `1` | 启动时执行一次 dummy 推理预热 |
| `LA_PORT` | `7860` | Gradio 服务端口 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像 (直连 huggingface.co 不可达) |
| `HF_HOME` | `/home/dgx/locate-thing/hf_cache` | 模型缓存目录 (避开 root 权限的默认目录) |

### 配置组合推荐

| 场景 | 配置 |
|------|------|
| 日常使用 (推荐) | `./run.sh` (默认 L0+L1+warmup) |
| 调试 / 排查问题 | `LA_COMPILE=0 LA_WARMUP=0 ./run.sh` |
| 极致速度 (精度略降) | `LA_FP8=1 ./run.sh` |
| 纯净模式 (最接近原始) | `LA_COMPILE=0 LA_TF32=0 LA_CUDNN_BENCH=0 LA_WARMUP=0 ./run.sh` |

## 优化层级与性能数据

### Benchmark 结果 (GB10, sweet.jpg, "dessert, cake")

| 配置 | tok/s | Prefill | 等效单图* |
|------|-------|---------|-----------|
| Baseline (无优化) | 48.5 | 1.18s | ~4.0s |
| L0 only (TF32/cuDNN) | 62.1 | 0.28s | ~3.2s |
| L0 + L1 (compile) | 68.0 | 0.29s | ~2.9s |

*等效单图 = 归一化到相同输出量 (~200 tokens, ~32 boxes)

### 各层级说明

**L0 — GPU 后端优化** (`LA_TF32`, `LA_CUDNN_BENCH`)

4 行全局设置，零风险。Prefill 从 1.18s 降至 0.28s (4.2x)，
因为 cuDNN benchmark 为 vision encoder 的卷积和 attention 自动选择最优 kernel。
Decode 吞吐 +28%，来自 TF32 矩阵乘。

**L1 — torch.compile** (`LA_COMPILE`)

`torch.compile(mode="reduce-overhead")` 做 kernel fusion + CUDA Graph capture。
Decode 吞吐在 L0 基础上再 +10%。代价是首次推理需 57 秒编译。
MTP decode 循环中的 Python 层 token 解码逻辑无法被 compile 覆盖，
所以 L1 的增益上限有限。

**L2 — FP8 量化** (`LA_FP8`) [实验性]

将 LLM 的 BF16 权重量化为 float8_e4m3fn。GB10 的 FP8 算力是 BF16 的 2.49x。
由于 364 GB/s 的 LPDDR5x 带宽远低于 HBM，权重读取减半的收益极大。
当前实现为 per-tensor 量化 (最简方案)，精度可能有微小变化。
默认关闭，需要手动 `LA_FP8=1` 开启。

## 架构说明

```
User Input (Image + Text Prompt)
        |
        v
+------------------+
|  Vision Encoder  |  MoonViT-SO-400M (27 layers, patch=14)
+------------------+
        |
        v
+------------------+
|   LLM Backbone   |  Qwen2.5-3B-Instruct (36 layers, GQA 8:1)
|   MTP Decode     |  Multi-Token Prediction (block_size=6)
+------------------+
        |
        v
+------------------+
|  Post-Processing |  Parse <ref>/<box> tokens -> pixel coords
+------------------+
        |
        v
  Output: Image with bounding boxes + JSON metadata
```

生成模式 (MTP) 是核心创新点：每步预测 6 个 token (block_size=6)，
通过 top-k 加权解码边界框坐标，只有异常时 fallback 到 AR (自回归)。
`hybrid` 模式在 MTP 速度和 AR 精度之间动态切换。

## 项目结构

```
/home/dgx/locate-anything/
  LocateAnything/           # 项目代码
    app.py                  # 主应用 (Gradio + FastAPI 后端)
    run.sh                  # 启动脚本 (配置全部环境变量)
    spaces.py               # HF ZeroGPU 本地 stub
    decord.py               # decord aarch64 stub
    index.html              # 前端页面
    requirements.txt        # Python 依赖
    assets/                 # 示例图片 + 字体
  hf_cache/                 # 模型缓存 (~7.3GB)
    hub/
      models--nvidia--LocateAnything-3B/
  OPTIMIZATION.md           # 完整优化设计文档
```

## 常见问题

**Q: 首次启动很慢？**
A: torch.compile 首次编译约需 57 秒，加上模型加载和 warmup 预热。
编译结果会被 PyTorch 缓存，后续重启编译时间会大幅缩短。

**Q: 如何关闭所有优化排查问题？**
A: `LA_COMPILE=0 LA_TF32=0 LA_CUDNN_BENCH=0 LA_WARMUP=0 ./run.sh`

**Q: huggingface.co 连不上？**
A: 本环境 huggingface.co 直连不通，已配置 hf-mirror.com 镜像。
模型缓存目录设在 `/home/dgx/locate-anything/hf_cache/`，
避开 root 权限的默认 `~/.cache/huggingface/`。

**Q: 视频推理不能用？**
A: aarch64 上没有 decord 的预编译 wheel，已用 stub 替代。
图像推理完全不受影响。如需视频功能，需要从源码编译 decord。

**Q: FP8 量化后精度如何？**
A: float8_e4m3fn 的动态范围有限 (正负448)。当前实现是 per-tensor 全量化，
适合验证速度。生产环境建议用 torchao 做 per-channel 量化或排除
LayerNorm/RMSNorm 等敏感层。

---

# NVIDIA Spark Conda 隔离环境与 Isaac Lab 安装指南

这份 README 整理自 `Spark_Conda隔离环境配置.md` 和 `install_isaaclab_spark_conda.sh`，用于在 NVIDIA Spark 上规划 LeRobot / ACT 训练环境，并在独立 Conda 环境中尝试安装 Isaac Sim 5.1 + Isaac Lab。

## 核心原则

不要把 LeRobot 训练依赖和 Isaac Lab 仿真依赖混在同一个 Python 环境里。推荐至少拆成两个 Conda 环境：

```text
lerobot-act  用于 LeRobot 数据集、ACT 训练、静态推理和模型导出
isaaclab     用于 Isaac Sim / Isaac Lab 仿真闭环
```

更稳的工作方式是：

```text
NVIDIA Spark:
  conda env: lerobot-act
  负责数据处理、ACT 训练、模型推理和模型导出

Isaac Lab 工作站:
  conda env: isaaclab
  负责 Isaac Sim / Isaac Lab 闭环验证
```

这样可以避免 PyTorch、CUDA、Isaac Sim 等版本互相污染，也方便把训练好的 `pretrained_model` 迁移到 Isaac Lab 机器。

## 目录文件

```text
7.6/
  README.md                         本文档
  install_isaaclab_spark_conda.sh   Isaac Sim 5.1 + Isaac Lab Conda 安装脚本
```

## 1. 先检查 Spark 系统

登录 Spark 后先执行：

```bash
uname -m
nvidia-smi
python3 --version
conda --version
```

重点确认：

```text
uname -m     是否为 aarch64 / arm64
nvidia-smi  GPU 是否可见
python3     系统 Python 版本
conda       Conda 是否可用
```

如果 `conda` 不存在，优先安装 Miniforge / Mambaforge。Spark 很可能是 ARM64 / aarch64 环境，不建议直接安装 x86 版 Anaconda。

## 2. 创建 LeRobot / ACT 训练环境

优先使用 Python 3.11：

```bash
conda create -n lerobot-act python=3.11 -y
conda activate lerobot-act
python -m pip install --upgrade pip setuptools wheel
```

先检查当前环境是否已经有可用 PyTorch：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

如果 Spark 官方镜像已经预装了适配 GPU 的 PyTorch，优先使用官方环境。如果需要手动安装，再按当前系统架构和 CUDA 版本选择对应 wheel，不要盲目照搬普通 x86 CUDA wheel。

## 3. 安装 LeRobot

在 `lerobot-act` 环境中执行：

```bash
conda activate lerobot-act
pip install "lerobot[dataset]"
```

如果 CLI 不可用，再从源码安装：

```bash
pip install git+https://github.com/huggingface/lerobot.git
```

验证：

```bash
python - <<'PY'
import lerobot
print("lerobot:", lerobot.__version__)
PY

which lerobot-train
```

## 4. 推荐工作目录

在 Spark 上建立固定工作目录：

```bash
mkdir -p ~/robot-act/{data,models,outputs,logs,src}
```

推荐布局：

```text
~/robot-act/
  data/     LeRobot 本地数据集或缓存
  models/   训练完成后的稳定模型目录
  outputs/  lerobot-train 输出目录
  logs/     训练日志
  src/      检查脚本和推理脚本
```

模型可以导出到：

```text
~/robot-act/models/thanos_model/
  config.json
  model.safetensors
  train_config.json
  policy_preprocessor.json
  policy_postprocessor.json
```

## 5. ACT 训练命令示例

```bash
conda activate lerobot-act

lerobot-train \
  --dataset.repo_id=imstevenpmwork/thanos_picking_power_gem \
  --policy.type=act \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.dim_model=512 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=4 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=2048 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=IMAGENET1K_V1 \
  --policy.dropout=0.1 \
  --policy.use_amp=true \
  --batch_size=32 \
  --steps=8000 \
  --save_freq=2000 \
  --env_eval_freq=2000 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=$HOME/robot-act/outputs/train/thanos_power_gem_act \
  --job_name=thanos_power_gem_act \
  --num_workers=4 \
  --wandb.enable=false \
  --policy.push_to_hub=false
```

如果显存或内存吃紧，先把 batch size 改为 16；仍然不稳定再改为 8。

## 6. 导出稳定模型

训练完成后找到最新 checkpoint：

```text
~/robot-act/outputs/train/thanos_power_gem_act/checkpoints/<step>/pretrained_model
```

复制到稳定目录：

```bash
mkdir -p ~/robot-act/models/thanos_model
cp -r ~/robot-act/outputs/train/thanos_power_gem_act/checkpoints/<step>/pretrained_model/* \
  ~/robot-act/models/thanos_model/
```

确认模型文件：

```bash
ls -lh ~/robot-act/models/thanos_model
```

至少应包含：

```text
config.json
model.safetensors
train_config.json
policy_preprocessor.json
policy_postprocessor.json
```

## 7. 静态推理验证

在 Spark 上先确认模型可以加载：

```bash
conda activate lerobot-act

python - <<'PY'
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("~/robot-act/models/thanos_model")
policy.eval()
print("ACT policy loaded")
PY
```

静态推理通过后，再进入 Isaac Lab 做闭环仿真验证。

## 8. 安装 Isaac Sim 5.1 + Isaac Lab

脚本 `install_isaaclab_spark_conda.sh` 会创建独立 Conda 环境、安装 Isaac Sim 5.1、安装 PyTorch、拉取 IsaacLab，并执行基础 import 验证。

默认运行：

```bash
bash install_isaaclab_spark_conda.sh
```

可选环境变量：

```bash
ENV_NAME=isaaclab \
WORKDIR=$HOME/robot-act/sim \
ISAACLAB_BRANCH=main \
INSTALL_FRAMEWORKS=none \
bash install_isaaclab_spark_conda.sh
```

参数说明：

```text
ENV_NAME            Conda 环境名，默认 isaaclab
WORKDIR             IsaacLab 克隆和安装目录，默认 $HOME/robot-act/sim
ISAACLAB_BRANCH     IsaacLab 分支，默认 main
INSTALL_FRAMEWORKS  Isaac Lab 额外强化学习框架，支持 none / all / rl_games / rsl_rl / sb3 / skrl / robomimic
```

安装完成后重新激活环境：

```bash
conda activate isaaclab
```

运行 headless smoke test：

```bash
cd ~/robot-act/sim/IsaacLab
OMNI_KIT_ACCEPT_EULA=YES python scripts/tutorials/00_sim/create_empty.py --headless
```

## 9. Spark 上的 Isaac Lab 风险

如果 Isaac Sim 在 Spark 上无法安装或启动，不要强行混装到 `lerobot-act` 环境。更推荐的分工是：

```text
Spark 负责训练和模型导出
x86_64 NVIDIA 工作站负责 Isaac Lab 闭环仿真
```

## 10. 从 Spark 迁移模型到 Isaac Lab

在 Spark 上打包：

```bash
cd ~/robot-act/models
tar -czf thanos_model.tar.gz thanos_model
```

拷贝到 Isaac Lab 机器：

```bash
scp ~/robot-act/models/thanos_model.tar.gz user@isaaclab-host:/home/user/models/
```

在 Isaac Lab 机器上解压：

```bash
cd /home/user/models
tar -xzf thanos_model.tar.gz
```

Isaac Lab 侧加载：

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("/home/user/models/thanos_model")
policy.eval()
```

## 11. Isaac Lab 侧映射重点

模型迁移过去后，最关键的是 observation / action 字段映射：

```text
Isaac Lab front camera  -> observation.images.front
Isaac Lab eagle camera  -> observation.images.eagle
Isaac Lab glove camera  -> observation.images.glove
Isaac Lab joint states  -> observation.state
LeRobot policy action   -> Isaac Lab joint / gripper controller
```

SO-101 关节顺序必须保持：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

迁移前必须检查：

```text
相机不是黑屏
图像 shape 正确
关节单位是弧度
关节顺序一致
夹爪开合方向一致
policy 输出 action 是 6 维
```

## 推荐最终工作流

```text
Spark / conda: lerobot-act
  1. 准备 LeRobot 数据集
  2. 训练 ACT
  3. 静态推理验证
  4. 导出 ~/robot-act/models/<task>_model

Isaac Lab / conda: isaaclab
  5. 解压模型
  6. 加载 ACTPolicy
  7. 编写 observation/action 映射
  8. 跑随机动作关节检查
  9. 跑闭环评估
  10. 根据失败样本补采或微调
```

一句话总结：用 Conda 把 Spark 上的训练环境和 Isaac Lab 仿真环境隔离开；Spark 先稳定负责训练和模型导出，Isaac Lab 再负责闭环验证。
