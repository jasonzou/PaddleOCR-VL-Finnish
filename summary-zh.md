# PaddleOCR-VL-Finnish — 项目概述

### 概述
**PaddleOCR-VL-Finnish** 使用 **LoRA**（rank-8）监督微调技术对 **PaddleOCR-VL-1.5**（一个视觉-语言 OCR 模型）进行芬兰语文字识别微调，项目面向 **Paddle 全球模型衍生赛**。

---

## 1. 数据准备

通过以下方式准备数据：

- 合成数据
- 搜索公开数据集
- 搜索芬兰语 OCR 学术论文，识别论文中使用的数据集并获取

### 1.1 合成数据

在 Huggingface 上筛选了流行芬兰字体以及排名前十的芬兰语数据集。将随机选择的芬兰语字体应用于从这些数据集中提取的内容，生成合成数据集，用于第一阶段微调。此阶段的目的是增强模型对芬兰语特殊字母（å, ä, ö）的识别能力。

### 1.2 搜索公开数据集

发现了 Kansallisarkisto 的 AIDA_ocr_training_data 数据集（https://huggingface.co/datasets/Kansallisarkisto/AIDA_ocr_training_data）。然而该数据集创建时间较早，包含合成数据、打字数据和手写数据。此外，该数据集的格式需要针对本项目进行调整。我移除了合成数据并修改了格式，然后在 Huggingface 上创建了三个数据集：
- https://huggingface.co/datasets/caveman273/aida-handwritten
- https://huggingface.co/datasets/caveman273/aida-typewritten
- https://huggingface.co/datasets/caveman273/aida-ship-info

### 1.3 搜索芬兰语 OCR 学术论文

对芬兰语 OCR 学术论文进行了文献检索。从多篇论文中识别出五个数据集。清理并格式化后，在 Huggingface 上创建了以下数据集：
- https://huggingface.co/datasets/caveman273/fin-13k
- https://huggingface.co/datasets/caveman273/swe-11k

另外两个可从以下来源下载：
- https://github.com/sdrobac/nodalida2017
- https://digi.kansalliskirjasto.fi/opendata/submit

`build_ocr_dataset.py` 可以生成与 Huggingface 数据集兼容的数据文件。

### 1.4 其他数据来源

调研了芬兰语语言资源库。选择了 theseui.fi（芬兰大学论文及学位论文库）作为另一个数据源，该网站拥有超过 34 万份 PDF 文件，且相对容易将 PDF 转换为 OCR 数据集。

该网站支持 OAI-PMH 协议，可通过 OAI-PMH 下载 PDF 文件。

### 1.5 数据集汇总

每个数据集的格式为 `{ "text", "image", "file_name" }`。

| 数据集 | 描述 |
|---------- |----------------|
| Synth-Font Scale | 50 k 条记录 |
| typewritten | 31.3 k 条记录 |
| handwritten | 9.36 k 条记录 |
| ship-info | 4.74 k 条记录 |
| fin-13k | 13 k 条记录 |
| swe-11k | 11 k 条记录 |
| theseus | 50 k 条记录 |
| digi-data |12.4 k 记录数 |
| natlib-data | 54 k  记录数 |
| nlf ocr groud truth | 500 k记录数 |

## 2. 微调

基于 PaddleFormers 框架，使用 **6 阶段课程学习**对 PaddleOCR-VL-1.5 进行 LoRA 微调。芬兰语受原生支持（111 种语言），无需修改分词器。

### 2.1 框架配置

| 配置项 | 值 |
|--------|-------|
| 序列长度 | 16384 |
| 无填充 | True（NaViT 动态分辨率） |
| LoRA rank | 8（默认；各阶段不同：16 → 8 → 4）|
| 优化器 | AdamW β1=0.9, β2=0.95, wd=0.1 |
| 混合精度 | bf16 |
| 评估指标 | NED（归一化编辑距离） |
| 模板 | `paddleocr_vl_v15` + 自定义插件 |

### 2.2 多阶段课程学习

| 阶段 | 数据集 | 概率 | 轮次 | 学习率 | 续自 |
|-------|-----------|------|--------|--------------|-------------|
| 0 – 合成预热 | synth-fonts (50k) | 1.0 | 3 | 1e-4 | PaddleOCR-VL-1.5 |
| 1 – 基础阶段 | hf (57k) | 1.0 | 5 | 5e-4 | stage0_synth |
| 2 – Digi-Natlib | digi-natlib + hf | 0.8, 0.2 | 3 | 2.5e-4 | stage1_hf |
| 3 – NLF | nlf + hf + digi-natlib | 0.5, 0.25, 0.25 | 3 | 1e-4 | stage2_digi |
| 4 – Theseus 页面 | theseus + hf + digi-natlib + nlf | 0.7, 0.1, 0.1, 0.1 | 3 | 5e-5 | stage3_nlf |
| 5 – 均衡混合 | hf + digi-natlib + nlf + theseus | 0.25 each | 4 | 5e-5 | stage4_theseus |

**概率设计逻辑：**
- **阶段 0–1**：单一数据集，无需重放（纯基础/合成）
- **阶段 2**：80/20 分配——优先新领域（digi-natlib），同时重放 hf 防止灾难性遗忘
- **阶段 3**：50/25/25——nlf 是大型新领域（328k），占主导权重；均衡重放保持先前领域性能
- **阶段 4**：70/10/10/10——theseus 是新领域（页面/区块 vs 文本行），重点学习并少量重放保留先前知识
- **阶段 5**：25/25/25/25——最终均衡混合，以同等权重协调所有领域

### 2.3 目标 NED 值

| 领域 | 目标 NED |
|--------|-----------|
| hf | < 0.05 |
| digi-natlib | < 0.08 |
| nlf | < 0.10 |
| theseus | < 0.12 |

### 2.5 微调参数

| 阶段 | LoRA rank | LoRA alpha | 学习率 | 轮次 | 设计逻辑 |
|-------|-----------|------------|--------------|--------|-----------|
| 0 – 合成预热 | 16 | 16 | 1e-4 | 3 | 高 rank 在干净合成数据上快速适应芬兰语特殊字母（å, ä, ö） |
| 1 – 基础阶段 | 8 | 16 | 5e-4 | 5 | 预热后降低 rank；更高学习率实现快速领域适应 |
| 2 – Digi-Natlib | 8 | 8 | 2.5e-4 | 3 | 降低 alpha/rank 比（1:1），对噪声真实文档进行保守更新 |
| 3 – NLF | 8 | 8 | 1e-4 | 3 | 保持稳定配置；适配器已捕获芬兰语模式，降低学习率 |
| 4 – Theseus 页面 | 4 | 8 | 5e-5 | 3 | 小领域使用最小 rank；最低学习率保留先前知识 |
| 5 – 均衡混合 | 4 | 8 | 5e-5 | 4 | 最终协调；均衡混合需要温和更新 |

**共享训练参数：**

| 参数 | 值 | 设计逻辑 |
|-----------|-------|-----------|
| 优化器 | AdamW β1=0.9, β2=0.95 | 标准 Transformer 微调 |
| 权重衰减 | 0.1 | 正则化 LoRA 矩阵；防止适配器值过大 |
| 预热比例 | 0.1 | 稳定早期训练；防止梯度突变 |
| 学习率调度器 | Cosine, min_lr=5e-5 | 平滑衰减；min_lr 下限防止后期权重剧烈变化 |
| Adamε | 1e-8 | 梯度更新的数值稳定性 |
| 混合精度 | bf16 | 匹配基础模型精度；降低显存 |
| 随机种子 | 23 | 确保可复现性 |

**为何逐阶段降低 rank 和学习率？**
- 早期阶段需要更高容量来学习芬兰语字形和文本行几何
- 后期阶段专注于适应新领域（digi-natlib、nlf、theseus）——适配器已捕获芬兰语模式，所需容量更小
- 后期阶段降低 rank 和学习率可减少对早期领域知识的灾难性遗忘

### 3. 训练数据构建严谨性

- **采集流程**：数据来源可追溯至 HuggingFace（`caveman273/aida-*`）和 Nodalida 2017 竞赛（GitHub `sdrobac/nodalida2017`）；提供可复现脚本（`build_ocr_dataset.py`）
- **标注完整性**：已记录数据模式（`file_name`、`text`、`image`）；管道输出 parquet → JSONL 并嵌入图片
- **质量控制**：通过 `check_dataset.py` 检查数据集；使用评估指标（NED、CER、WER）验证模型质量
- **数据分析**：由 `check_dataset.py` 生成统计信息；README 中提供 base vs. LoRA 性能对比

### 4. 微调流水线中的图像增强

PaddleOCR-VL-1.5 使用 **NaViT 风格动态分辨率视觉编码器**——无需手动调整图像大小。模型通过 `get_smarted_resize` 自动将图像调整为最佳倍数 patch 大小尺寸。

**增强架构**（`vl_v15_template.py` 中的 `PaddleOCRVLV15Plugin`）：

两种模式——**经典模式**（逐条 `p` 门控）和**策略驱动模式**（双层采样）：

1. **策略门控**：决定应用多少以及哪些类别的变换
2. **资格门控**：每条目的 `p` 字段决定其是否为候选

**类别权重**（策略模式）：region_mask: 0.15, noise: 0.25, distortion: 0.20, degradation: 0.20, geometric: 0.10, padding: 0.10。比例可在 `data/aug_config.yaml` 中调整。

**可用的增强类别：**

| 类别 | 变换 |
|----------|-----------|
| **噪声（全图）** | `GaussianNoise`、`JpegCompression`、`AlbumentationsISONoise` |
| **模糊（全图）** | `GaussianBlur`、`AlbumentationsMotionBlur` |
| **颜色** | `ColorJitter`（亮度/对比度/饱和度/色相） |
| **畸变** | `Curve`（正弦波）、`AlbumentationsGridDistortion`、`AlbumentationsOpticalDistortion` |
| **丢弃** | `DropoutHorizontal`、`DropoutVertical`、`DropoutRand` |
| **区域掩码** | `MaskRandomBlock`、`MaskRandomPixel`、`MaskGaussianNoise`、`MaskSaltPepper`、`MaskBlur`、`MaskMosaic` |
| **几何** | `ImagePadding`、`LineOverlay`、`TextBorder` |

**序列长度过滤**：由于 max_seq_len=16384，超宽图像会产生过多视觉 token。每个样本应用 **1200 字符** 限制以防止 token 溢出。超过限制的样本被丢弃（从不截断真实标签）。

### 5. 初步演示

- 模型已发布至 Huggingface：https://huggingface.co/caveman273/ppocrvl-v1.5-finnish
- 演示站点托管于 Huggingface：https://caveman273-paddleocrvl-v1-5-finnish.hf.space/