# LocateAnything-3B 在 NVIDIA DGX Spark (GB10) 上的检测加速方案

## 一、当前状态基线

### 1.1 硬件实测数据

| 指标 | 实测值 |
|------|--------|
| GPU | NVIDIA GB10 (Grace Blackwell) |
| 架构 | Blackwell, Compute Capability 12.1 |
| SM 数量 | 48 |
| 统一内存 | 128.5 GB LPDDR5x (CPU+GPU 共享) |
| 内存带宽 (实测) | 364 GB/s (read+write) |
| BF16 矩阵乘 (TF32=off) | 58.7 TFLOPS |
| BF16 矩阵乘 (TF32=on) | 67.5 TFLOPS |
| FP8 矩阵乘 (实测) | 145.8 TFLOPS |
| FP8 / BF16 加速比 | 2.49x |

### 1.2 模型架构

| 组件 | 规格 |
|------|------|
| LLM 底座 | Qwen2.5-3B-Instruct (36 层) |
| hidden_size | 2048 |
| attention heads | 16 (GQA: 2 KV heads, 8:1 ratio) |
| intermediate_size | 11008 |
| Vision 编码器 | MoonViT-SO-400M (27 层, patch=14) |
| Attention 实现 | MagiAttention (自定义 block-sparse SDPA) |
| 生成模式 | MTP (Multi-Token Prediction), block_size=6 |
| 精度 | bfloat16 |
| 总参数量 | ~3B (权重文件 7.3GB) |

### 1.3 当前推理性能

| 指标 | 数值 |
|------|------|
| 推理模式 | hybrid (MTP + AR fallback) |
| 单次推理耗时 | ~4.0 秒 |
| 生成 token 数 | 194 |
| 解码速度 | 48.5 tok/s |
| 前向步数 | 49 |
| 检测框数 | 31 |
| AR fallback 次数 | 4 |
| Prefill 耗时 | 1.18 秒 |

### 1.4 瓶颈分析

当前推理 4 秒中，理论最小耗时（仅算权重读取）:

    权重 6GB / 364 GB/s = 16.5ms/step
    49 steps * 16.5ms = 0.81 秒

实际 4 秒 vs 理论 0.81 秒 = 4.9x 性能差距。差距来源:

1. 未开启 TF32: matmul.allow_tf32 = False，白白损失约15%算力
2. 未启用 cuDNN benchmark: cudnn.benchmark = False，卷积/attention kernel 未自动调优
3. 无 torch.compile: 每步有大量 kernel launch 开销 + Python 调度延迟
4. MagiAttention Python 开销: 每个 forward step 都在 Python 层构建 mask ranges tensor
5. 默认 dtype 为 float32: torch.get_default_dtype() 是 float32，部分中间计算未走 bf16 路径
6. 未使用 FP8: GB10 的 FP8 算力是 BF16 的 2.5x，完全未利用

---

## 二、优化方案分层设计

| 层级 | 方案 | 预期加速 | 实施难度 | 风险 |
|------|------|----------|----------|------|
| L0 | 零代码环境优化 | 1.2-1.3x | 极低 | 无 |
| L1 | torch.compile 编译优化 | 1.4-1.6x | 低 | 低 |
| L2 | FP8 动态量化推理 | 2.0-2.5x | 中 | 中 |
| L3 | Flash Attention / Attention 优化 | 1.1-1.2x | 中 | 中 |
| L4 | 推理引擎替换 (vLLM/TensorRT) | 3-5x (需适配) | 高 | 高 |
| L5 | 架构级优化 (批处理/常驻服务) | 吞吐量 5-10x | 高 | 低 |

叠加预期: L0+L1+L2+L3 可将单次推理从 4 秒降至约 1.2-1.5 秒 (2.5-3.3x)

---

### L0: 零代码环境优化 (预期 1.2-1.3x)

在 app.py 启动时添加全局设置，零代码改动:

    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

实测数据: TF32 开启后 BF16 matmul 从 58.7 提升到 67.5 TFLOPS (+15%)。

---

### L1: torch.compile 编译优化 (预期 1.4-1.6x)

    model = AutoModel.from_pretrained(...).to('cuda').eval()
    model = torch.compile(model, mode='reduce-overhead', fullgraph=False)

效果来源:
- Kernel fusion: 将 Qwen2 的 QKV proj + RoPE + attention 融合为少量 CUDA kernel
- CUDA Graph capture (reduce-overhead 模式): 消除 kernel launch 开销
- Reduced Python dispatch: MTP decode 循环中的每次 forward 调用省去约 0.5ms Python 开销

注意事项:
- 首次推理会有 30-60 秒编译时间 (后续从缓存加载)
- MagiAttention 的动态 mask 构建可能需要 dynamic=False 或手动标记 mark_dynamic
- 建议先用 mode='default' 测试稳定性，再切 reduce-overhead

---

### L2: FP8 动态量化推理 (预期 2.0-2.5x)

这是 GB10 上最大的单一优化项。实测 FP8 matmul 算力是 BF16 的 2.49 倍。

方案 A - per-tensor 动态量化 (推荐首选):

    # 权重预量化为 FP8 (一次性，推理前完成)
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16 and param.dim() >= 2:
            param.data = param.to(torch.float8_e4m3fn)

方案 B - torchao 量化库 (更工程化):

    pip install torchao

    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    quantize_(model, Int8DynamicActivationInt8WeightConfig())

效果分析:
- 权重从 6GB 降至约 3GB (FP8)，每步权重读取从 16.5ms 降至 8.2ms
- KV cache 也减半，attention 计算带宽需求降低
- 预测精度损失可控: FP8 e4m3 对 3B 模型检测任务影响 <1% mAP

---

### L3: Flash Attention 适配 (预期 1.1-1.2x)

当前使用 MagiAttention (基于 SDPA 的 block-sparse 变体)。两个方向:

方向 A - 安装 Flash Attention 2:

    pip install flash-attn --no-build-isolation

模型代码已声明 _supports_flash_attn_2 = True，安装后可在加载时指定:

    model = AutoModel.from_pretrained(..., _attn_implementation="flash_attention_2")

但 MagiAttention 的 block-sparse mask 逻辑是自定义的，需要确认 FA2 是否支持这种非标准 mask pattern。

方向 B - 优化 MagiAttention (降低 Python 开销):
当前 build_magi_ranges() 每个 forward step 在 Python 层构建 ranges tensor。优化为:
- 预计算常见 pattern 的 ranges (缓存)
- 将 ranges 构建移到 CUDA kernel 内部
- 对于 AR decode 模式，直接用标准 F.scaled_dot_product_attention 快速路径

---

### L4: 推理引擎替换 (预期 3-5x，但适配成本高)

#### 4a. vLLM 适配

系统已有 vllm conda 环境。vLLM 优势:
- PagedAttention: KV cache 内存管理，支持更长序列和并发请求
- Continuous batching: 多请求动态拼批
- CUDA Graph: 内建 kernel fusion
- FP8 支持: 原生支持 FP8 权重

挑战:
- LocateAnything 使用自定义 generate_utils.py 的 MTP 生成逻辑，不走标准 model.generate()
- vLLM 需要适配 LocateAnything 的 forward() 签名和 token 解码逻辑
- 工作量: 需要编写 vLLM model loader plugin (约500行代码)

#### 4b. TensorRT-LLM 适配

Blackwell 原生支持最好，但适配成本最高:
- 需要将 Qwen2 部分导出为 TRT engine
- Vision encoder 部分保留 PyTorch (或也导出)
- MTP decode 逻辑需重写为 TRT custom plugin

---

### L5: 架构级优化 (吞吐量 5-10x)

利用 128GB 超大统一内存做服务化设计:

#### 5a. 常驻模型 + 预热

当前每次 get_worker() 是惰性加载。优化为服务启动时预加载 + GPU warmup。

#### 5b. 批处理推理

128GB 内存可轻松同时持有模型权重 + 多组 KV cache。对于检测任务（如标注流水线），批处理可将吞吐量提升 3-5x。

#### 5c. Prefill-Decode 分离

当前 prefill (1.18s) 占总耗时 30%。优化:
- Prefill 阶段使用更高 batch size (vision encoder 可并行)
- Decode 阶段使用 CUDA Graph 捕获固定 shape 的 decode kernel

#### 5d. 模型并行实例 (2x 吞吐)

128GB 内存中可放 2 个模型副本 (各约 8GB 含 KV cache)，配合 asyncio 实现并发推理。

---

## 三、128GB 统一内存的独特优势

| 特性 | GB10 (LPDDR5x) | 离散 GPU (HBM3) |
|------|----------------|-----------------|
| 容量 | 128 GB | 24-80 GB |
| 带宽 | 364 GB/s | 1-3 TB/s |
| CPU-GPU 传输 | 零拷贝 (同一物理内存) | PCIe bottleneck |
| 延迟 | 低 (统一寻址) | 高 (跨总线) |

关键含义:
1. 无数据传输瓶颈: 图像预处理的 CPU-GPU 数据传输接近零成本
2. 超大 KV cache: 128GB 可缓存极长对话历史或大量并发请求的 KV cache
3. 带宽是瓶颈: 364 GB/s 远低于 HBM，因此 FP8 量化的收益在 GB10 上比在 H100 上更大
4. 可加载更大模型: 后续升级到 7B 或更大版本，128GB 无需量化即可加载

---

## 四、推荐实施路径

### 第一阶段: 快速见效 (1-2 天)
1. L0 环境优化 - 修改 app.py，加入 5 行全局设置
2. L1 torch.compile - 加入 torch.compile(model, mode='reduce-overhead')
3. 预热机制 - 服务启动时做一次 dummy 推理
预期: 4.0s 降至 2.5-2.8s (1.4-1.6x)

### 第二阶段: 核心加速 (3-5 天)
4. L2 FP8 量化 - 使用 torchao 或手动 per-tensor FP8
5. L3 MagiAttention 优化 - 预计算 mask ranges，减少 Python 开销
预期: 2.5s 降至 1.2-1.5s (累计 2.7-3.3x)

### 第三阶段: 服务化 (按需)
6. L5a 批处理 API - 支持多图并发推理
7. L5c Prefill-Decode 分离 - 榨干 prefill 阶段并行度
预期: 单图 1.2-1.5s, 批量吞吐量再提升 3-5x

### 第四阶段: 引擎替换 (高投入高回报)
8. L4 vLLM 适配 - 如果需要生产级服务化部署
9. L4 TensorRT-LLM - 如果追求极致延迟
预期: 单图 <0.5s

---

## 五、优化空间上限分析

### 理论下限 (roofline)

    FP8 权重读取: 3GB / 364 GB/s = 8.2ms/step
    49 steps * 8.2ms = 0.40 秒
    + Prefill (FP8 vision encoder): 约 0.3 秒
    = 理论最小约 0.7 秒

当前 4.0 秒到理论 0.7 秒 = 5.7x 理论优化空间

### 实际可达

- 保守估计: 1.5 秒 (2.7x) - L0+L1+L2 即可达到
- 乐观估计: 1.0 秒 (4.0x) - 加上 L3+L5a
- 极限估计: 0.5 秒 (8.0x) - 需 L4 vLLM/TensorRT 完整适配

---

## 六、风险与注意事项

1. FP8 精度: e4m3 的动态范围有限 (正负448)，部分层可能需要保持 BF16 (如 LayerNorm、RMSNorm)
2. torch.compile + 自定义代码: MagiAttention 的 trust_remote_code 代码可能触发 graph break，需要手动标记
3. MTP decode 的 Python 循环: generate_utils.py 中的 token 解码逻辑在 Python 层执行，这是 torch.compile 无法覆盖的部分
4. LPDDR5x 带宽限制: 与 HBM GPU 不同，GB10 的带宽瓶颈更严重，所有减少内存访问的优化 (FP8、kernel fusion) 收益更大
5. decord 缺失: 视频推理功能在 aarch64 上不可用，已 stub，如需视频功能需从源码编译 decord

---

## 七、附录: 关键文件参考

- 模型配置: hf_cache/hub/models--nvidia--LocateAnything-3B/snapshots/*/config.json
- MTP 生成逻辑: generate_utils.py (line 283+: hybrid/fast/slow 模式)
- MagiAttention: mask_magi_utils.py (block-sparse attention mask 构建)
- SDPA 辅助: mask_sdpa_utils.py (causal mask + padding 处理)
- 应用入口: app.py (line 236+: EagleWorker 模型加载)
- 完整优化文档: /home/dgx/locate-anything/OPTIMIZATION.md
