# Analysis of Bias in Paper Review

## 项目简介 / Project Overview

本项目旨在通过自然语言处理和大语言模型技术，量化分析学术论文审稿过程中的偏差现象。核心假设：即使不同审稿人发现了相同的优缺点，给出的最终分数也可能存在显著差异。

This project aims to quantitatively analyze bias in academic paper reviews using natural language processing and large language models. Core hypothesis: Even when different reviewers identify the same strengths and weaknesses, their final scores may differ significantly.

## 仓库结构 / Repository Structure

- **bias/**: 偏差分析框架和工具 / Bias analysis framework and tools
  - 包含用于分析审稿过程偏差的 Python 脚本
  - 包括数据加载器、特征提取器和可视化工具
  - 详细使用说明请参见 `bias/GETTING_STARTED.md`

- **ICLR_2025_CLEAN/**: ICLR 2025 会议论文数据集 / ICLR 2025 conference paper dataset
  - ICLR 2025 提交论文的清洁数据集
  - 按论文组织，包含审稿意见和元数据

## 核心流程（四步骤）/ Core Workflow (4 Steps)

本系统采用四步骤流程进行偏差分析：

### 步骤1: 特征提取 (Feature Extraction)
- 使用 LLM 独立提取每个审稿人的优缺点
- 输出文件: `results/extraction/extraction_results.json`

### 步骤2: 匿名化处理 (Anonymization)
- **代码逻辑处理**（非 LLM）
- 去除审稿人信息，只保留描述和类别
- 随机打乱顺序
- 输出文件: `results/anonymized/anonymized_pros_cons.json`

### 步骤3: 权重量化 (Weight Quantification)
- 基于匿名化的优缺点 + PDF论文全文内容
- 使用 LLM 为每个优缺点分配权重
- 输出文件: `results/quantified/quantified_weights.json`

### 步骤4: 匹配计算 (Matching & Calculation)
- **代码逻辑处理**（非 LLM）
- 根据映射文件匹配回对应的审稿人
- 线性相加计算每个审稿人的期望分数

```
期望分数 = 基准分数(5.0) + Σ(优点权重) + Σ(缺点权重)
偏差 = 实际分数 - 期望分数
```

## 安装 / Installation

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

# 4. 配置 API 密钥
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

## 快速开始 / Quick Start

### 快速测试（2篇论文）
```bash
cd bias
python quick_test.py
```

### 完整分析
```bash
cd bias
python run_iclr_analysis.py
```

### 使用 Python API
```python
from bias.pipeline import ReviewBiasAnalysisPipeline

# 初始化 pipeline
pipeline = ReviewBiasAnalysisPipeline()

# 加载 ICLR 2025 数据
pipeline.load_from_openreview("../ICLR_2025_CLEAN")

# 限制测试数量（可选）
pipeline.papers = pipeline.papers[:2]

# 运行完整四步骤分析
results = pipeline.run_full_analysis()

# 或分步执行
pipeline.step1_extract_features()       # LLM提取
pipeline.step2_anonymize_and_shuffle()  # 代码处理
pipeline.step3_quantify_weights()       # LLM量化
pipeline.step4_match_and_calculate()    # 代码计算
```

## 输出文件 / Output Files

分析完成后，结果保存在 `results/` 目录：

```
results/
├── extraction/              # 步骤1: 原始提取结果
│   └── extraction_results.json
├── anonymized/              # 步骤2: 匿名化后的数据
│   ├── anonymized_pros_cons.json
│   └── original_mapping.json
├── quantified/              # 步骤3: 量化结果
│   └── quantified_weights.json
├── paper_details/           # 每篇论文的详细报告
│   └── {paper_id}_details.md
├── figures/                 # 可视化图表
├── analysis_results.json    # 完整分析结果
└── analysis_report.txt      # 文本报告
```

## 核心算法 / Core Algorithm

### 期望分数计算公式

```
Expected_Score = Base_Score + Σ(Pros_Weights) + Σ(Cons_Weights)
```

其中：
- `Base_Score`: 基准分数 (默认 5.0)
- `Pros_Weights`: 优点权重（正值，如 +0.5 到 +2.0）
- `Cons_Weights`: 缺点权重（负值，如 -0.5 到 -2.0）

### 偏差计算

```
Bias = Actual_Score - Expected_Score
```

- 正偏差：审稿人给分偏高
- 负偏差：审稿人给分偏低
- 零偏差：给分合理

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
- PDF 论文全文

## 注意事项 / Notes

### API 费用
- 本系统调用 OpenAI API，会产生费用
- 默认使用 `gpt-4o-mini` 模型（成本较低）
- 系统自动启用缓存，避免重复调用
- 建议先用 `quick_test.py` 测试（仅2篇论文）

### 处理时间
- 特征提取：约 5-10 秒/审稿
- 权重量化：约 5-10 秒/论文
- 2篇论文测试：约 2-5 分钟
- 完整数据集：约 30-60 分钟

## License

MIT License
