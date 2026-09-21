# Colibrì 全模型转换与使用指南

> 从 7B 到 2.8T，九大模型家族的完整落地手册

Colibrì 是一个零依赖纯 C 实现的 MoE 推理引擎，核心设计理念是**将 VRAM、RAM 与 NVMe 视为统一的推理层级**——高速内存不足只影响速度，不改变模型语义。目前支持九个模型家族，每个家族一个 C 文件，共用同一套 `coli chat` / `coli serve` / `coli web` 前端。

本文将从实战角度，逐一讲解每个模型的**获取、转换、构建与运行**全流程，并给出硬件选型建议。

---

## 目录

- [环境准备](#环境准备)
- [模型总览](#模型总览)
- [一、OLMoE — 7B 入门首选](#一olmoe--7b-入门首选)
- [二、Qwen3.6 — 35B 性价比之王](#二qwen36--35b-性价比之王)
- [三、GLM-5.2/5.3 — 744B 参考模型](#三glm-5253--744b-参考模型)
- [四、GLM-5.3-Flash — 321B 含视觉](#四glm-53-flash--321b-含视觉)
- [五、DeepSeek V4 Flash — 284B 原生 FP4](#五deepseek-v4-flash--284b-原生-fp4)
- [六、DeepSeek V4.1 Flash — 552B 含视觉](#六deepseek-v41-flash--552b-含视觉)
- [七、Qwen3.8-Flash-Next — 176B 纯 CPU](#七qwen38-flash-next--176b-纯-cpu)
- [八、Inkling — 975B 多模态](#八inkling--975b-多模态)
- [九、Kimi K3 — 2.8T 超大模型](#九kimi-k3--28t-超大模型)
- [通用运行命令](#通用运行命令)
- [硬件选型速查表](#硬件选型速查表)
- [常见问题](#常见问题)

---

## 环境准备

### 构建引擎

```bash
git clone https://github.com/JustVugg/colibri && cd colibri/c
./setup.sh    # 检查 gcc/OpenMP、构建并运行自测
```

### Python 依赖（仅转换时需要）

```bash
pip install torch>=2.4 safetensors>=0.4 huggingface-hub>=0.24 numpy>=1.26
```

> 引擎本身是纯 C 二进制，运行时**不需要 Python**。Python 仅在模型转换阶段使用。

### 中国大陆网络加速

HuggingFace 直链通常不可达，两种替代方案：

```bash
# 方案一：HF-Mirror 镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1    # 实测 22-46 MB/s

# 方案二：ModelScope（国内 CDN 通常 50-100+ MB/s）
pip install modelscope
modelscope download --model <model_id>
```

---

## 模型总览

| 模型家族 | 总参数 / 激活参数 | 磁盘占用 | 最低 RAM | 需要转换？ | 引擎 |
|---|---|---|---|---|---|
| **OLMoE** | 7B / 1B | ~7 GB | 8 GB | ✅ merged int8 | `olmoe` |
| **Qwen3.6** | 35B / 3B | ~20 GB | 24 GB | ✅ int4-gs64 | `qwen36` |
| **GLM-5.2** | 744B / 40B | ~372 GB | 16 GB | ✅ int4-gs64 + int8 MTP | `colibri` |
| **GLM-5.3** | 744B / 40B | ~419 GB | 16 GB | ✅ int4-gs64 | `colibri` |
| **GLM-5.3-Flash** | 321B / 40B | ~195 GB | 25 GB | ✅ int4-gs64 + BF16 dense | `glm53` |
| **DeepSeek V4 Flash** | 284B / 13B | ~167 GB | 16 GB | ❌ 原生 FP4 直读 | `deepseek_v4` |
| **DeepSeek V4.1 Flash** | 552B / 16B | ~203 GB | 16 GB | ❌ 原生 FP4 直读 | `deepseek_v41` |
| **Qwen3.8-Flash-Next** | 176B / 6B | ~185 GB | 16 GB | ❌ 原生 block-FP8 直读 | `qwen38` |
| **Inkling** | 975B / 41B | ~469 GB | 25 GB | ✅ int4 experts + BF16 dense | `inkling` |
| **Kimi K3** | 2.8T / 104B | ~1.6 TB | 32 GB | ❌ 原生 MXFP4 直读 | `kimi_k3` |

> **关键洞察**：Colibrì 的核心优势在于**专家从磁盘流式加载**。你不需要把整个模型装进 RAM——只需存放稠密部分（~10-25 GB），路由专家按需从 NVMe 读取。磁盘速度决定推理速度。

---

## 一、OLMoE — 7B 入门首选

**最适合**：磁盘空间有限、RAM 8-16 GB 的用户，快速体验 MoE 推理。

### 获取与转换

OLMoE 没有预转换容器，需要用 `convert_olmoe_merged.py` 将原始 HuggingFace checkpoint 转换为 merged int8 格式。转换脚本会将每个 expert 的 `gate_proj` + `up_proj` + `down_proj` 合并为单个 `merged_weight`，使引擎只需一次磁盘读取即可加载一个 expert。

```bash
# 方式一：从 HuggingFace 流式下载+转换（推荐，峰值磁盘仅需一个 shard）
python3 tools/convert_olmoe_merged.py \
  --repo allenai/OLMoE-1B-7B-0125-Instruct \
  --out /path/on/NVMe/olmoe_i8

# 方式二：从本地已下载的 checkpoint 转换
# 先下载（ModelScope 或 HuggingFace）
modelscope download --model allenai/OLMoE-1B-7B-0125-Instruct
# 再转换
python3 tools/convert_olmoe_merged.py \
  --model /path/to/OLMoE-1B-7B-0125-Instruct \
  --out /path/on/NVMe/olmoe_i8
```

> ⚠️ **必须使用 Instruct 版本**（`0125-Instruct`），不要用基座模型（`0924`）。基座模型未经指令微调，只会重复输入。

### 构建与运行

```bash
make -C c olmoe
COLI_MODEL=/path/on/NVMe/olmoe_i8 ./coli chat
```

### 特点

- CPU-only，无 GPU 后端
- 64 experts/layer，每 token 激活 8 个
- 上下文窗口 4096 tokens
- 不支持 tool calling、推测解码

---

## 二、Qwen3.6 — 35B 性价比之王

**最适合**：消费级硬件（24 GB RAM），需要高质量中文/英文输出。

### 获取（预转换容器，推荐）

```bash
# 推荐：int4-gs64 容器（量化误差比 per-row 低 44%）
python3 -m pip install -U "huggingface_hub[cli]"
hf_download Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64 --local-dir /path/on/NVMe/qwen36_i4
```

### 自行转换

```bash
python3 tools/convert_qwen36.py \
  --repo Qwen/Qwen3.6-35B-A3B \
  --out /path/on/NVMe/qwen36_i4 \
  --ebits 4 --group-size 64
```

`--ebits` 参数控制 expert 量化位数（2-8），默认 4（int4-gs64）。`--ebits 8` 为 int8 参考锚点。

### 构建与运行

```bash
# CPU only
make -C c qwen36

# 可选 CUDA VRAM expert tier（实测 1.44 → 10.05 tok/s，7.0× 加速）
make -C c qwen36 CUDA=1

COLI_MODEL=/path/on/NVMe/qwen36_i4 ./coli chat
```

### 特点

- **混合架构**：Gated Attention + Gated DeltaNet（线性注意力），每层同时包含两种注意力机制
- 256 experts/layer，每 token 激活 8 个
- 上下文窗口 262144 tokens
- 支持 CUDA VRAM expert tier
- 同一引擎也支持 Qwen3.8-2.4T-A95B（92 层 × 512 experts，自动识别）

---

## 三、GLM-5.2/5.3 — 744B 参考模型

**最适合**：大内存主机（32 GB+），追求最强中文推理能力。

### 获取（预转换容器）

```bash
# GLM-5.2：含 int8 MTP head 的 gs64 容器（372 GB）
hf_download mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp \
  --local-dir /path/on/NVMe/glm52_i4

# GLM-5.3：gs64 容器，不含 MTP head（419 GB）
hf_download Justvugg/GLM-5.3-colibri-int4-g64 \
  --local-dir /path/on/NVMe/glm53_i4
```

> ⚠️ **不要使用旧的 per-row int4 镜像**（`mateogrgic/…`、`jlnsrk/…`）：质量实测低约 9 个百分点，也是 think-mode 循环的根因。MTP head 必须是 **int8**（int4 的草稿接受率为 0%）。

### 自行从 FP8 源转换

```bash
./coli convert --model /path/on/NVMe/glm52_i4
# 逐 shard 下载并转换，可断点续传，峰值磁盘 = 1 shard + 输出
```

或直接调用转换脚本：

```bash
python3 tools/convert_fp8_to_int4.py \
  --repo zai-org/GLM-5.2-FP8 \
  --outdir /path/on/NVMe/glm52_i4
```

### 构建与运行

```bash
make -C c colibri    # 或 make -C c glm
COLI_MODEL=/path/on/NVMe/glm52_i4 ./coli chat
```

### 特点

- 78 层 × 256 experts，每 token 激活约 40B 参数
- MLA 注意力（KV 状态缩小 57×）
- DSA 稀疏注意力（lightning indexer）
- 原生 MTP 推测解码（GLM-5.2，需 int8 MTP head）
- 支持 tool calling
- 上下文窗口 1048576 tokens

---

## 四、GLM-5.3-Flash — 321B 含视觉

**最适合**：需要视觉理解能力，25 GB+ RAM 的用户。

### 获取与转换

无预转换容器，需从 HuggingFace 源转换：

```bash
# 流式下载+转换（推荐）
python3 tools/convert_glm53.py \
  --outdir /path/on/NVMe/glm53_flash_i4 \
  --min-free-gb 30

# 从本地已下载的 checkpoint 转换
python3 tools/convert_glm53.py \
  --indir /path/to/GLM-5.3-Flash \
  --outdir /path/on/NVMe/glm53_flash_i4
```

转换策略：routed experts → int4-gs64，dense/MLA/KDA/vision → BF16（精度由引擎在加载时选择）。

### 构建与运行

```bash
make -C c glm53
COLI_MODEL=/path/on/NVMe/glm53_flash_i4 ./coli chat
```

### 特点

- 含视觉能力（vision tower）
- dense 权重保持 BF16，精度可由引擎在加载时动态选择
- 支持 tool calling
- 上下文窗口 1048576 tokens

---

## 五、DeepSeek V4 Flash — 284B 原生 FP4

**最适合**：希望零转换直接使用的用户，可选 CUDA 加速。

### 获取

**无需转换**——引擎直接流式读取官方 checkpoint，routed experts 保持原生 FP4，dense 保持 fp8-e4m3。

```bash
# 下载官方 checkpoint
hf_download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir /path/on/NVMe/DeepSeek-V4-Flash

# 或使用 REAP 裁剪版（150B，仅 85 GB，132/256 experts）
hf_download puwaer/DeepSeek-V4-Flash-0731-reap-150b --local-dir /path/on/NVMe/DeepSeek-V4-Flash-REAP
```

### 构建与运行

```bash
# CPU
make -C c deepseek-v4

# 可选 CUDA（GTX 10 系及以上，RTX 50 最佳）
make -C c deepseek-v4 CUDA=1

COLI_MODEL=/path/on/NVMe/DeepSeek-V4-Flash ./coli chat --ram 32
```

### 特点

- **零转换**：直接读取官方 FP4/fp8 checkpoint
- 43 层 × 256 experts，top-k 6
- REAP 裁剪版（150B）使用同一引擎，无需额外操作
- 可选 CUDA 层级（prefill 5-10×，decode ~2.5× 加速）
- 支持 tool calling
- 上下文窗口 1048576 tokens

---

## 六、DeepSeek V4.1 Flash — 552B 含视觉

**最适合**：需要视觉+tool calling+推测解码的全功能模型。

### 获取

**无需转换**——与 V4 相同，直接读取官方 checkpoint。

```bash
hf_download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /path/on/NVMe/DeepSeek-V4.1-Flash
```

可选：生成 engram 索引文件（加速 n-gram memory 查找）：

```bash
python3 tools/prepare_dsv41.py --model /path/on/NVMe/DeepSeek-V4.1-Flash
```

### 构建与运行

```bash
make -C c deepseek_v41    # CPU only（无 GPU 后端）
COLI_MODEL=/path/on/NVMe/DeepSeek-V4.1-Flash ./coli chat --ram 32
```

### 特点

- **零转换**：experts 原生 fp4，dense 原生 fp8-e4m3
- 40 层 × 384 experts，top-6
- 含视觉能力、tool calling、DSpark 推测解码 head
- 203 GB n-gram memory 从磁盘按需读取（每 token 仅几百字节）
- 每 token 路由专家仅 4.5 GB（vs GLM-5.2 的 12.7 GB）
- CPU-only，无 GPU 后端

---

## 七、Qwen3.8-Flash-Next — 176B 纯 CPU

**最适合**：纯 CPU 环境，16-24 GB RAM。

### 获取

**无需转换**——直接读取官方 FP8 checkpoint。

```bash
hf_download Qwen/Qwen3.8-Flash-Next-FP8 --local-dir /path/on/NVMe/Qwen38-FP8
```

### 构建与运行

```bash
make -C c qwen38    # CPU only
COLI_MODEL=/path/on/NVMe/Qwen38-FP8 ./coli chat
```

### 特点

- 125B 参数 + 51B n-gram memory
- experts 保持原生 block-FP8，PLE 可分页
- 支持 tool calling
- CPU-only，无 GPU 后端
- 上下文窗口 262144 tokens

---

## 八、Inkling — 975B 多模态

**最适合**：大内存主机（25 GB+），需要 Thinking Machines 的多模态能力。

### 获取（预转换容器）

```bash
hf_download nbeerbower/Inkling-colibri-int4 \
  --local-dir /path/on/NVMe/inkling_i4
```

### 自行转换

```bash
# 从本地 BF16 checkpoint 转换
python3 tools/convert_inkling_int4.py \
  --indir /path/to/Inkling-BF16 \
  --outdir /path/on/NVMe/inkling_i4

# 可选：边下载边转换（--watch 模式）
python3 tools/convert_inkling_int4.py \
  --indir /path/to/Inkling-downloading \
  --outdir /path/on/NVMe/inkling_i4 \
  --watch
```

### RAM 不足时的优化

Inkling 的 dense 权重为 BF16（49.4 GB resident），RAM 不足时可用专用工具转为 int4：

```bash
# 先估算
python3 tools/convert_inkling_dense_int4.py --dir /path/on/NVMe/inkling_i4 --plan
# 确认后转换（dense 从 49.4 GB → 15.3 GB，可在 25 GB 机器上运行）
python3 tools/convert_inkling_dense_int4.py --dir /path/on/NVMe/inkling_i4
```

### 构建与运行

```bash
make -C c inkling
COLI_MODEL=/path/on/NVMe/inkling_i4 ./coli chat
```

### 特点

- 含音频能力（audio tower）
- int4 experts + BF16 dense（或 int4 dense）
- 不支持 tool calling
- 上下文窗口 1048576 tokens

---

## 九、Kimi K3 — 2.8T 超大模型

**最适合**：大内存服务器（32 GB+），超大上下文需求。

### 获取

**无需转换**——引擎直接流式读取官方 checkpoint，routed experts 保持原生 MXFP4（QAT 训练精度），BF16 dense 在加载时量化。

```bash
hf_download moonshotai/Kimi-K3 --local-dir /path/on/NVMe/Kimi-K3
```

### 可选：repack 优化

虽然不需要转换，但可以用 `k3_repack.py` 重新打包以优化加载速度——将 experts 按 ID 顺序排列、六个张量连续存储（一次 pread 读取一个 expert），并在拷贝过程中量化 BF16 dense 权重：

```bash
python3 tools/k3_repack.py \
  /path/on/NVMe/Kimi-K3 \
  /path/on/NVMe/Kimi-K3-repacked \
  --bits 8    # dense 量化位数（默认 int8）
```

### 构建与运行

```bash
make -C c kimi_k3
COLI_MODEL=/path/on/NVMe/Kimi-K3 ./coli chat --ram 32
```

### 特点

- **2.8T 参数**，当前支持的最大模型
- 原生 MXFP4 experts（QAT 训练精度，零量化损失）
- KDA + MLA 双路径注意力
- 支持 tool calling
- 可选 recurrent-state checkpoints（`COLI_K3_CKPT=N`）实现长会话暖启
- 可选 Vulkan expert tier（`K3_VK=auto`）
- 上下文窗口 1048576 tokens

---

## 通用运行命令

所有模型共享同一套 CLI，`coli` 启动器会根据模型的 `config.json` 自动选择对应引擎：

```bash
# 交互式聊天
COLI_MODEL=/path/to/model ./coli chat

# OpenAI 兼容 API 服务
COLI_MODEL=/path/to/model ./coli serve

# API + 网页仪表盘
COLI_MODEL=/path/to/model ./coli web

# 一次性生成
COLI_MODEL=/path/to/model ./coli run "你的提示词"

# 查看资源规划
COLI_MODEL=/path/to/model ./coli plan

# 就绪检查
COLI_MODEL=/path/to/model ./coli doctor

# 性能调优（测量并保存最快执行配置）
COLI_MODEL=/path/to/model ./coli tune
```

### 常用运行参数

| 参数 | 说明 | 示例 |
|---|---|---|
| `--ram N` | RAM 预算（GB） | `--ram 24` |
| `--ngen N` | 最大生成 token 数 | `--ngen 512` |
| `--topp P` | 采样 top-p | `--topp 0.9` |
| `--cap N` | 每层缓存槽位数 | `--cap 64` |
| `--attach URL` | 连接已运行的 serve 实例 | `--attach http://127.0.0.1:8000` |

---

## 硬件选型速查表

| 你的硬件 | 推荐模型 | 预期速度 |
|---|---|---|
| 8 GB RAM, 50 GB 磁盘 | OLMoE (7B) | ~0.2 tok/s (NVMe) |
| 16 GB RAM, 200 GB 磁盘 | DeepSeek V4 REAP 150B / Qwen3.8 | ~1-3 tok/s |
| 24 GB RAM, 25 GB 磁盘 | Qwen3.6 (35B) | ~2-5 tok/s |
| 24 GB RAM, 400 GB 磁盘 | GLM-5.2 (744B) | ~1-3 tok/s |
| 32 GB RAM, 500 GB 磁盘 | Inkling (975B) / DeepSeek V4.1 | ~1-2 tok/s |
| 64 GB+ RAM, 1.6 TB 磁盘 | Kimi K3 (2.8T) | ~0.5-1 tok/s |

> **核心原则**：模型必须放在**快速本地存储（NVMe）**上。网络挂载或慢速 HDD 会使推理速度严重下降。磁盘 I/O 延迟是 MoE 流式推理的瓶颈。

---

## 常见问题

### Q: 下载的原始 HuggingFace 模型能直接用吗？

只有以下三个模型**无需转换**，引擎直接读取官方 checkpoint：
- **DeepSeek V4 Flash / V4.1 Flash**（原生 FP4/fp8）
- **Qwen3.8-Flash-Next**（原生 block-FP8）
- **Kimi K3**（原生 MXFP4）

其余模型都需要先转换为 colibri 容器格式。

### Q: OLMoE 输出重复/循环怎么办？

你下载的是基座模型（`OLMoE-1B-7B-0924`），需要使用 **Instruct 版本**（`OLMoE-1B-7B-0125-Instruct`）并运行 `convert_olmoe_merged.py` 转换。

### Q: GLM-5.2 的 MTP head 为什么必须是 int8？

int4 量化的 MTP head 草稿接受率会崩塌到 0-4%，完全无法加速推理。int8 head 的接受率正常，可实现每次 forward 产生 2.2-2.8 个 token。

### Q: 转换可以中断续跑吗？

所有转换脚本都支持断点续传：
- `convert_fp8_to_int4.py`：逐 shard 下载+转换+删除源 shard
- `convert_olmoe_merged.py`：扫描已有输出 shard，跳过已完成的张量
- `convert_qwen36.py`：同上
- `convert_glm53.py`：同上
- `k3_repack.py`：每个 shard 写入 .tmp 后 rename，中断后重跑自动续传

### Q: 如何在 MacPorts 环境下使用 OpenMP？

Colibrì 的 `setup.sh` 和 `Makefile` 已支持 MacPorts 的 libomp 检测回退（`/opt/local/include/libomp/omp.h`），无需额外配置。如果 OpenMP 缺失，引擎会以单线程模式构建（速度较慢）。

### Q: 可以把构建好的 colibri 拷贝到其他机器使用吗？

可以，但注意：
- 引擎以 `ARCH=native` 编译，**仅适用于相同架构**（x86→x86，ARM→ARM）
- 使用 `make install` 安装到 `/usr/local/bin/coli`，即可随处使用
- 或手动拷贝：将 `coli`（启动器）、引擎二进制、`tools/` 目录和 Python 支持模块放在同一目录

### Q: 如何选择 int4 还是 int8？

- **int4-gs64**（group-scaled，group_size=64）：磁盘占用减半，量化误差比 per-row int4 低 44%，**推荐**
- **int8**：精度更高，作为参考锚点或对精度极度敏感的场景
- **per-row int4**：旧格式，质量差 ~9pp，**不推荐**

---

> 本文基于 colibri v1.11.0+，项目持续迭代中，请以 [GitHub 仓库](https://github.com/tekintian/colibri) 最新版本为准。
> website: [https://ai.tekin.cn](https://ai.tekin.cn)
> email: [tekintian@gmail.com](mailto:tekintian@gmail.com)
> QQ: 932256355
