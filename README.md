# Analysis of Bias in Paper Review

## 项目简介 / Project Overview

本项目旨在通过自然语言处理和大语言模型技术，量化分析学术论文审稿过程中的偏差现象。核心假设：即使不同审稿人发现了相同的优缺点，给出的最终分数也可能存在显著差异。

This project aims to quantitatively analyze bias in academic paper reviews using natural language processing and large language models. Core hypothesis: Even when different reviewers identify the same strengths and weaknesses, their final scores may differ significantly.

## 仓库结构 / Repository Structure

- **bias/**: 偏差分析框架和工具 / Bias analysis framework and tools
  - 包含用于分析审稿过程偏差的 Python 脚本
  - 包括数据加载器、特征提取器和可视化工具
  - 详细使用说明请参见 `bias/README.md`

- **ICLR_2025_CLEAN/**: ICLR 2025 会议论文数据集 / ICLR 2025 conference paper dataset
  - ICLR 2025 提交论文的清洁数据集
  - 按论文组织，包含审稿意见和元数据

## 核心功能 / Core Features

1. **数据预处理 / Data Preprocessing**: 支持 JSON/CSV 格式的论文和审稿数据
2. **特征提取 / Feature Extraction**: 使用 LLM 从审稿意见中提取结构化的优缺点
3. **分值量化 / Score Quantification**: 使用 LLM 为每个优缺点赋予量化权重
4. **偏差分析 / Bias Analysis**: 计算期望分数与实际分数的差异
5. **可视化 / Visualization**: 生成统计图表展示偏差分布

## 安装步骤 / Installation

```bash
# 1. 克隆仓库
git clone git@github.com:wutaghost/Analysis-of-bias-in-paper-review.git
cd Analysis-of-bias-in-paper-review

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
cd bias
pip install -r requirements.txt
```

## 快速开始 / Quick Start

```python
from bias.pipeline import ReviewBiasAnalysisPipeline

# 初始化 pipeline
pipeline = ReviewBiasAnalysisPipeline()

# 加载 ICLR 2025 数据
pipeline.load_data("ICLR_2025_CLEAN/", format="iclr")

# 运行完整分析
results = pipeline.run_full_analysis()

# 生成可视化报告
pipeline.generate_visualizations(output_dir="results")
```

## 核心算法 / Core Algorithm

### 期望分数计算公式

```
Expected_Score = Σ(Pros_Weights) - Σ(Cons_Weights) + Base_Score
```

### 偏差计算

```
Bias = Actual_Score - Expected_Score
```

## 文档 / Documentation

详细文档请参见：
- `bias/GETTING_STARTED.md` - 入门指南
- `bias/PROJECT_SUMMARY.md` - 项目总结  
- `bias/QUICK_REFERENCE.md` - 快速参考
- `bias/使用说明.txt` - 中文使用说明

## 数据集说明 / Dataset Description

ICLR 2025 数据集包含 200+ 篇论文的提交信息和审稿意见，每个论文文件夹包含：
- 论文元数据
- 审稿意见
- 评分信息
- 讨论记录

## 许可证 / License

MIT License

