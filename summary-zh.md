# 芬兰语 OCR

## 概述
**PaddleOCR-VL-Finnish** 是在 **PaddleOCR-VL-1.5**（视觉-语言 OCR 模型）上使用 **LoRA** 微调和 **DPO**（直接偏好优化）进行芬兰语文本识别的模型。

---

## 1. 数据准备

我们通过以下方式准备数据：

- 合成数据
  - 搜索开放数据集
  - 搜索芬兰语 OCR 研究论文，识别论文中使用的数据集并获取
  - 评估数据集
  - 选择数据集
  - 清洗/转换数据集

### 1.0 数据集选择理由

| 数据源 | 理由 |
|--------|------|
| **AIDA（手写、打字、船舶信息）** | 来自 Kansallisarkisto（芬兰国家档案馆）的最大开源芬兰语 OCR 数据集。涵盖三种不同的视觉领域——手写历史文档、打字记录和船舶货单表格——均来自同一来源。原始数据集还包含合成数据，已被移除，因为我们使用更好的字体控制生成自己的合成数据。 |
| **fin-13k / swe-11k** | 通过芬兰语 OCR 论文的文献综述发现。fin-13k 增加了额外的真实芬兰语文本行多样性。swe-11k（瑞典语）被包含是因为瑞典语是芬兰的第二官方语言，共享 ä/ö 变音符号，且视觉领域（历史报纸）与芬兰语数据集互补。 |
| **Digi-Natlib (nodalida2017)** | 来自 Nodalida 2017 竞赛——在芬兰语 OCR 研究论文中被广泛引用。包含数字化芬兰报纸（digi，约 12k 条记录）和国家图书馆目录卡片（natlib，约 54k 条记录），每条都有对齐的二值化图像和标注文本。一种来源中两种不同的视觉风格（新闻栏 vs. 图书馆卡片）。 |
| **Theseus** | Theseus.fi 是所有芬兰应用科学大学的中央论文/学位论文库，拥有 34 万+ PDF 文件。选择它而非其他芬兰语言银行来源，是因为它支持 OAI-PMH 协议进行系统化采集，且 PDF 页面图像渲染简单直接。提供学术芬兰语散文，以页面和段落粒度呈现——补充了以上主要为文本行级别的数据集。 |
| **NLF OCR 标注数据** | 芬兰国家图书馆 OCR 标注数据集（约 50 万条记录），来自 digi.kansalliskirjasto.fi/opendata。包含芬兰语和瑞典语文档，带有 ALTO XML 标注（词级边界框）和高分辨率 TIFF 扫描。多栏报纸版面（1 栏到 4 栏）提供了文本行级别数据集所缺乏的版面阅读多样性。 |

更重要的是，我们使用了一些额外的机制来确保数据集的质量：
1. 对每个数据集，随机选择 100 条记录，并手动检查所选记录，直到所有记录均无问题。
2. 对于 theseus，我们使用两个 OCR 模型（mineru 和 glm-ocr）辅助验证。仅当提取的内容与 mineru 和 glm-ocr 的输出匹配时，该内容才会被包含。虽然这种方式会忽略一些更困难的情况，但数据集的质量得到了保证。

### 流程 - 搜索开放数据集

https://huggingface.co/datasets/Kansallisarkisto/AIDA_ocr_training_data 数据集是几年前创建的。数据集包含合成数据、打字数据和手写数据。此外，数据格式需要为本项目更改。我移除了合成数据并更改了格式，然后在 Huggingface 上创建了三个数据集：
- https://huggingface.co/datasets/caveman273/aida-handwritten
- https://huggingface.co/datasets/caveman273/aida-typewritten
- https://huggingface.co/datasets/caveman273/aida-ship-info

**选择理由：** AIDA 是来自国家档案馆的最大开源芬兰语 OCR 数据集。其三个子组件涵盖不同的视觉领域（手写、打字、表格数据），提供互补的训练信号。

**清洗方法：**
- 移除原始 AIDA 的合成子组件（与我们自己的合成数据流水线重复）。
- Schema 规范化为 `{text, image, file_name}`——列名在常见变体（text/transcript/label/ground_truth, image/img/image_path, file_name/filename/id）中自动检测。
- 跳过不存在的图像路径和非字符串文本条目。
- 所有图像规范化为 PNG 字节以确保一致的序列化。

### 流程 - 搜索芬兰语 OCR 研究论文

开展了芬兰语 OCR 研究论文的文献搜索。从各种论文中，确定了五个数据集。清洗和格式化后，我在 Huggingface 上创建了以下内容：
- https://huggingface.co/datasets/caveman273/fin-13k
- https://huggingface.co/datasets/caveman273/swe-11k
另外两个可以从以下位置下载：
- https://github.com/sdrobac/nodalida2017
- https://digi.kansalliskirjasto.fi/opendata/submit

**选择理由：** 芬兰语 OCR 学术论文引用这些数据集作为基准。fin-13k 提供额外的真实芬兰语文本行图像。swe-11k 涵盖瑞典语（芬兰的第二官方语言，共享 ä/ö 变音符号）。nodalida2017 数据集提供数字化芬兰报纸和国家图书馆目录卡片。NLF 标注数据（digi.kansalliskirjasto.fi）是可用的最大标注芬兰语 OCR 语料库。

**清洗方法：**
- fin-13k / swe-11k：重新格式化为 HF 数据集格式，具有一致的 {text, image, file_name} schema。使用与 AIDA 相同的列自动检测惯例。
- digi-natlib：空文本的二进制（.bin.png）/ 标注（.gt.txt）对被跳过。通过 `source__split__stem.png` 构造唯一文件名以避免跨目录名称冲突。
- NLF：详细清洗见下方流程部分。

### 流程 - 其他数据来源
考察了芬兰语言银行。选择 theseus.fi（芬兰大学的论文和学位论文库）作为另一个数据来源，因为该网站拥有超过 34 万个 PDF 文件，且将 PDF 文件转换为 OCR 数据集相对容易。

该网站支持 OAI-PMH 协议。PDF 文件可通过 OAI-PMH 下载。

**选择理由：** Theseus 提供学术芬兰语散文——这是以上主要为报纸/图书馆数据集所缺乏的文本领域。通过标准协议（OAI-PMH）可获取 34 万+ PDF，提供可扩展、可复现的数据获取。

**清洗方法：**
- 少于 2 行文本的页面 → 跳过（段落提取内容不足）。
- 少于 2 行文本的段落裁剪 → 跳过。
- 每页段落提取：保存整个页面图像（type=page），然后通过划分文本行区域创建最多 5 个段落裁剪（type=paragraph）。
- PDF 以 300 DPI 渲染以确保 OCR 质量分辨率。
- MuPDF 结构树诊断警告被抑制（对 Theseus PDF 无害但噪音大）。
- 通过 PDF 内部条目 ID 前缀防止文件名冲突。

### 流程 - NLF OCR 标注数据

NLF OCR 标注数据经过最彻底的清洗流水线：

- **无标注扫描**：没有对应 `-gt2.xml` 的 TIF 文件被静默跳过（数据集中存在未标注的扫描）。
- **页面级过滤**：ALTO XML 解析 `<Page>`、`<TextBlock>` 和 `<TextLine>` 元素。没有文本块或缺少尺寸的页面被跳过。
- **分栏检测**（`_nlf_detect_columns`）：
  - 宽度超过页面宽度 65% 的文本块被视为通栏（标题、大标题），分配到第 0 栏。
  - 小型装饰性文本块（页码、页眉）——定义为少于 2 行文本且高度小于页面的 5%——被排除在分栏边界检测之外，但仍分配到阅读顺序的栏中。
  - 当连续文本块 x 中心之间的间隙超过页面宽度的 12% 时，检测到栏的"中缝"边界。
  - 支持任意栏数：1 栏、2 栏、3 栏、4 栏报纸版面。
- **阅读顺序重建**：文本块按（栏 ID，然后从上到下的 y 位置）排序，确保多栏页面的正确文本流。
- **段落裁剪生成**：对每栏，将 1 到 N 个连续文本块的滑动窗口组合成段落裁剪。每个段落至少 20 个字符。每页去重完全相同的段落文本。每栏最多提取 10 个段落裁剪，以防止对视觉重复区域的过度采样。

## 数据集

| # | 数据集 | 来源 | 类型 | 训练集大小 | 测试集大小 | 描述 |
|---|--------|------|------|------------|------------|------|
| 1 | aida-typewritten | caveman273/aida-typewritten (HF) | 文本行 | 31,300 | 4,272 | AIDA（芬兰商业文档档案）中的打字商业文档。芬兰企业/行政的机械打字遗留文本。 |
| 2 | aida-handwritten | caveman273/aida-handwritten (HF) | 文本行 | 9,360 | 1,270 | AIDA 档案中的手写商业文档。各种纸张类型上的草书和手写芬兰语文本。 |
| 3 | aida-ship-info | caveman273/aida-ship-info (HF) | 文本行 | 4,740 | 472 | AIDA 档案中的船舶信息记录。带有打字的芬兰语文本的表格/半结构化航运记录。 |
| 4 | fin-13k | caveman273/fin-13k (HF) | 文本行 | 13,000 | 653 | 芬兰语文本图像数据集。从扫描文档中提取的多样化芬兰语文本行。 |
| 5 | swe-11k | caveman273/swe-11k (HF) | 文本行 | 11,000 | 556 | 瑞典语文本图像数据集。来自数字档案的瑞典语文本行，适用于双语芬兰语-瑞典语 OCR。 |
| 6 | digi-natlib | sdrobac/nodalida2017 (GitHub) | 文本行 | 66,400 | 3,327 | 数字化芬兰报纸（digi：12.4k）和国家图书馆目录卡片（natlib：54k）及其标注文本。作为 Nodalida 2017 的一部分发布。 |
| 7 | theseus | theseus.fi (OAI-PMH) | 页面 / 段落 | 44,500 | 2,230 | 通过 OAI-PMH 从 Theseus.fi 采集的芬兰学术论文。下载 PDF，以 300 DPI 渲染；通过 pdfplumber + PyMuPDF 提取整个页面图像 + 段落裁剪（每页最多 5 个）。 |
| 8 | nlf-ocr | 芬兰国家图书馆 | 页面 / 段落 | 21,200 | 1,079 | 来自 ALTO XML + TIFF 配对的 NLF OCR 标注数据。芬兰语（`nlf_fi`）和瑞典语（`nlf_sv`）集合。通过自动多栏版面检测从 ALTO TextBlock 标注中提取整个页面图像和段落级裁剪。 |

## 方法

多数据集 LoRA 微调，使用逆平方根频率采样。

### 逆平方根频率采样

不再简单地重复文件，而是在 DataLoader 层面定义采样权重：P(dataset) ∝ 1/√N。

这个公式是平衡"充分学习"与"防止过拟合"的黄金标准（用于 GPT 等多语言训练）。它避免了两种陷阱：
- **1/N 加权**：小数据集权重过高（→ 过拟合）
- **直接拼接**：大数据集淹没小数据集

8 个真实数据集的精确归一化权重（可直接填入配置）：

| 数据集 | 样本数 | 权重 (1/√N) | 归一化采样概率 |
|--------|--------|-------------|----------------|
| digi-natlib-gen  | 66,518 | 0.00388 | 6.0%  (原始占比 33.0%) |
| theseus-gen      | 44,591 | 0.00473 | 7.4%  (原始占比 22.1%) |
| aida_typewritten | 31,272 | 0.00566 | 8.8%  (原始占比 15.5%) |
| nlf-ocr          | 21,200 | 0.00687 | 10.7% (原始占比 10.5%) |
| aida_handwritten | 9,367  | 0.01033 | 16.1% (原始占比 4.6%)  |
| swe_11k          | 11,097 | 0.00950 | 14.8% (原始占比 5.5%)  |
| fin_13k          | 13,040 | 0.00876 | 13.6% (原始占比 6.5%)  |
| aida_ship_info   | 4,740  | 0.01453 | 22.6% (原始占比 2.4%)  |

**效果：** aida_ship_info 的曝光率从可怜的 2.4% 大幅提升至 22.6%。但因为是随机采样（有放回），模型每个 Epoch 看到的都是其原始 4.7k 张图像的重采样组合——绝不会像物理重复那样导致模型在几千张图上死记硬背（过拟合）。

### 训练配置

| 参数 | 值 | 理由 |
|------|-----|------|
| LoRA rank/alpha | 8/8 | 对噪声真实文档的保守更新 |
| 学习率 | 3e-4 | 适合多数据集微调 |
| 优化器 | AdamW β1=0.9, β2=0.95 | Transformer 微调的标准选择 |
| 权重衰减 | 0.1 | 正则化 LoRA 矩阵；防止适配器值过大 |
| 热身比例 | 0.1 | 稳定早期训练；防止梯度尖峰 |
| 学习率调度器 | Cosine, min_lr=5e-5 | 平滑衰减；min_lr 下限防止权重剧烈变化 |
| Adamε | 1e-8 | 梯度更新的数值稳定性 |
| 混合精度 | bf16 | 与基础模型精度匹配；减少内存 |
| 随机种子 | 23 | 跨运行可复现性 |

### 图像增强

PaddleOCR-VL-1.5 使用 **NaViT 风格的动态分辨率视觉编码器**——无需手动调整图像大小。模型通过 `get_smarted_resize` 自动将图像调整到最优的 patch 大小倍数尺寸。

**增强架构**（`vl_v15_template.py` 中的 `PaddleOCRVLV15Plugin`）：

两种模式——**经典**（每条数据的 `p` 门控）和**策略驱动**（两层采样）：

1. **策略门控**：决定应用*多少个*和*哪些类别*的变换
2. **资格门控**：每条数据自己的 `p` 字段决定其是否为候选对象

**类别权重**（策略模式）：region_mask: 0.15, noise: 0.25, distortion: 0.20, degradation: 0.20, geometric: 0.10, padding: 0.10。比例可在 `data/aug_config.yaml` 中调整。

**可用的增强类别：**

| 类别 | 变换 |
|------|------|
| **噪声（整图）** | `GaussianNoise`, `JpegCompression`, `AlbumentationsISONoise` |
| **模糊（整图）** | `GaussianBlur`, `AlbumentationsMotionBlur` |
| **颜色** | `ColorJitter`（亮度/对比度/饱和度/色调） |
| **变形** | `Curve`（正弦波）, `AlbumentationsGridDistortion`, `AlbumentationsOpticalDistortion` |
| **丢弃** | `DropoutHorizontal`, `DropoutVertical`, `DropoutRand` |
| **区域遮罩** | `MaskRandomBlock`, `MaskRandomPixel`, `MaskGaussianNoise`, `MaskSaltPepper`, `MaskBlur`, `MaskMosaic` |
| **几何** | `ImagePadding`, `LineOverlay`, `TextBorder` |

**序列长度过滤**：当 max_seq_len=16384 时，极宽的图像可能生成过多的视觉 token。每个样本应用 **1200 字符**的限制以防止 token 溢出。超过此限制的样本被丢弃（标注文本从不截断）。

## 结果
### NED
| 数据集 | 基线 | LoRA |
|--------|------|------|
| aida-typewritten  | 0.0530 | 0.0266 |
| aida-handwritten  | 0.3840 | 0.1412 |
| aida-ship-info    | 0.1538 | 0.0490 |
| fin-13k           | 0.1654 | 0.0606 |
| swe-11k           | 0.1389 | 0.0840 |
| digi-natlib       | 0.2201 | 0.1150 |
| theseus           | 0.6952 | 0.3408 |
| nlf-ocr           | 0.9037 | 0.4804 |

### 未来方向
对 DPO（直接偏好优化）进行了评估。由于数据集之间存在差异，构建 DPO 对具有挑战性。需要更多探索，因此目前未包含。

### 演示

- 模型已发布至 Huggingface：https://huggingface.co/caveman273/ppocrvl-v1.5-finnish
- 演示站点托管在 Huggingface：https://caveman273-paddleocrvl-v1-5-finnish.hf.space/
- 芬兰语 OCR 评估数据集（14k）已发布至 Huggingface：https://huggingface.co/datasets/caveman273/Finnish-OCR-evaluation
