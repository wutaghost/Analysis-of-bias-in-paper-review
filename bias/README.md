# 论文审稿偏差分析系统 (Review Bias Analysis Pipeline)

## 项目简介

本项目旨在通过自然语言处理和大语言模型技术，量化分析学术论文审稿过程中的偏差现象。核心假设：即使不同审稿人发现了相同的优缺点，给出的最终分数也可能存在显著差异。

## 核心功能

1. **数据预处理**：支持JSON/CSV格式的论文和审稿数据
2. **特征提取**：使用LLM从审稿意见中提取结构化的优缺点
3. **分值量化**：使用LLM为每个优缺点赋予量化权重
4. **偏差分析**：计算期望分数与实际分数的差异
5. **可视化**：生成统计图表展示偏差分布

## 安装步骤

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置API密钥
cp .env.example .env
# 编辑 .env 文件，填入你的 OpenAI API Key
```

## 数据格式

### 输入数据格式（JSON）

```json
[
  {
    "paper_id": "paper_001",
    "title": "论文标题",
    "abstract": "论文摘要...",
    "paper_content": "论文全文（可选）",
    "reviews": [
      {
        "reviewer_id": "reviewer_1",
        "review_text": "审稿意见全文...",
        "actual_score": 7
      },
      {
        "reviewer_id": "reviewer_2",
        "review_text": "审稿意见全文...",
        "actual_score": 5
      }
    ]
  }
]
```

### 输入数据格式（CSV）

CSV文件需要包含以下列：
- `paper_id`: 论文ID
- `title`: 论文标题
- `abstract`: 论文摘要
- `paper_content`: 论文全文（可选）
- `reviewer_id`: 审稿人ID
- `review_text`: 审稿意见
- `actual_score`: 实际打分

## 使用方法

### 快速开始

```python
from pipeline import ReviewBiasAnalysisPipeline

# 初始化pipeline
pipeline = ReviewBiasAnalysisPipeline()

# 加载数据
pipeline.load_data("data/reviews.json", format="json")

# 运行完整分析
results = pipeline.run_full_analysis()

# 生成可视化报告
pipeline.generate_visualizations(output_dir="results")
```

### 分步执行

```python
# 1. 提取优缺点
pipeline.extract_pros_cons()

# 2. 量化优缺点权重
pipeline.quantify_weights()

# 3. 计算偏差
bias_results = pipeline.analyze_bias()

# 4. 统计分析
stats = pipeline.statistical_analysis()

# 5. 生成报告
pipeline.generate_report("results/analysis_report.html")
```

## 项目结构

```
bias/
├── config.py               # 配置管理
├── data_loader.py          # 数据加载模块
├── feature_extractor.py    # 特征提取模块
├── llm_quantifier.py       # LLM分值量化模块
├── bias_analyzer.py        # 偏差分析模块
├── visualizer.py           # 可视化模块
├── pipeline.py             # 主pipeline
├── utils.py                # 工具函数
├── main.py                 # 入口脚本
├── requirements.txt        # 依赖列表
└── README.md              # 本文件
```

## 核心算法

### 期望分数计算公式

```
Expected_Score = Σ(Pros_Weights) - Σ(Cons_Weights) + Base_Score
```

### 偏差计算

```
Bias = Actual_Score - Expected_Score
```

## 注意事项

1. **API调用成本**：本系统会频繁调用LLM API，请注意控制成本
2. **缓存机制**：系统默认启用缓存，避免重复调用
3. **错误处理**：API调用失败会自动重试，最多3次
4. **并发控制**：为避免触发API速率限制，默认串行处理

## 许可证

MIT License




