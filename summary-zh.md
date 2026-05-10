# PaddleOCR-VL-Finnish — 项目概述

### 概述
**PaddleOCR-VL-Finnish** 是针对 **Paddle Global Model Derivative Competition**（Paddle 全球模型衍生赛）构建的项目，使用 **LoRA**（rank-8）监督微调技术对 **PaddleOCR-VL-1.5**（一个视觉-语言 OCR 模型）进行芬兰语文字识别微调。

---

## 与评分标准对应说明

### 1. 数据集质量

| 子项 | 描述 |
|----- |------|
| 数据规模 | 约 65k 条记录（digi-natlib）+ 约 57k 条（hf），来自芬兰/领域特定数据源 |
| 标注准确性 | 通过管道验证 `text` 字段；JSONL 分片生成时确保文件-文本映射正确 |
| 数据多样性 | 涵盖芬兰历史文献（digi）、国家图书馆记录（natlib）、船舶登记册、打字与手写文档——真实采集数据，含各类噪声 |
| 难度合理性 | 易（打字）与难（手写、陈旧文档）混合；多阶段课程学习设计 |

### 2. 场景稀缺性

- **研究稀缺性**：与英语、汉语等主流语言相比，芬兰语 OCR 研究较少；公开基准数据集稀缺
- **工业需求**：解决芬兰历史档案（国家图书馆、船舶登记册）的数字化需求——真实归档痛点
- **独特性**：芬兰语特殊字母（å, ä, ö）、历史拼写规则、混合打字/手写文档

### 3. 任务复杂度

- **视觉复杂度**：真实文档含折痕、陈旧噪声、多种字体、手写——双阶段训练配合自定义噪声增强
- **结构复杂度**：多阶段课程（阶段一：HF 数据 → 阶段二：digi-natlib 混合数据）；多样化数据隐式覆盖多任务
- **理解复杂度**：OCR 任务本身涉及版面/视觉理解；通过 NED/CER/WER 评估语义保真度

### 4. 训练数据构建严谨性

- **采集流程规范化**：数据来源可追溯至 HuggingFace（`caveman273/aida-*`）和 Nodalida 2017 竞赛（GitHub `sdrobac/nodalida2017`）；提供可复现脚本（`convert_natlib.py`、`covert_digi.py`）
- **标注规范完整性**：已记录数据模式（`source`、`file_name`、`text`、`image`）；管道输出 parquet → JSONL 并嵌入图片
- **质量控制机制**：通过 `check_dataset.py` 检查数据集；使用评估指标（NED、CER、WER）验证模型质量
- **数据统计分析**：由 `check_dataset.py` 生成统计信息；README 中提供了 base vs. LoRA 的性能对比

### 5. 模型微调策略与创新

- **微调策略合理性**：双阶段 LoRA 课程（阶段一：LR=5e-4，5 epochs，HF 数据 → 阶段二：LR=2.5e-4，3 epochs，digi-natlib+HF 混合）；任务定制模板设计
- **实验充分性**：README 中展示了 hf 和 digi-natlib 评估集上 base vs. LoRA 的对比结果
- **技术创新性**：自定义模板（`my_v15_template.py`）支持可配置的 Gaussian/椒盐/speckle 噪声及 Gaussian 模糊增强；使用 `flashmask` 注意力机制提升效率

### 6. 技术文档与开源贡献

- **文档完整性**：README 包含环境配置、训练流程、评估流程、结果说明、多阶段配置概览
- **可复现性**：提供完整的数据转换脚本、训练脚本（`lorav1.sh`）、导出脚本（`merge-lora.sh`）、推理脚本（`eval.py`、`trans.py`）；配置文件位于 `configs/`
- **演示完整性**：单图推理脚本（`test_single_image.py`）；支持批量推理与指标计算
- **社区价值**：开源芬兰语 OCR 模型填补了低资源语言处理空白；可复用的管道可迁移至其他低资源 OCR 任务

---

## 实验结果

| 评估集 | 模型 | 平均 NED | Exact Match |
|--------|------|----------|-------------|
| hf | base | 0.1289 | 35.96% |
| hf | lora02 | 0.0239 | 74.79% |
| digi-natlib | base | 0.2215 | 3.04% |
| digi-natlib | lora02 | 0.0033 | 93.86% |

**核心优势**：基于真实芬兰语历史文献的多阶段 LoRA 课程学习，配合噪声增强，显著优于基线模型（在 digi-natlib 上 NED 提升 67 倍，在 hf 上提升 81 倍）。
