# 论文审稿偏差分析系统 (Review Bias Analysis Pipeline)

## 项目简介

本项目旨在通过自然语言处理和大语言模型技术，量化分析学术论文审稿过程中的偏差现象。核心假设：即使不同审稿人发现了相同的优缺点，给出的最终分数也可能存在显著差异。

## 核心功能

1. **数据预处理**：支持JSON/CSV格式的论文和审稿数据
2. **特征提取**：使用LLM从审稿意见中提取结构化的优缺点
3. **分值量化**：使用LLM为每个优缺点赋予量化权重
4. **偏差分析**：计算期望分数与实际分数的差异
5. **可视化**：生成统计图表展示偏差分布

## 快速开始（3分钟）

### 1. 安装依赖

```bash
cd /path/to/bias
pip install -r requirements.txt
```

### 2. 配置API密钥

**方式A：使用环境变量（推荐）**
```bash
export OPENAI_API_KEY='your-api-key-here'
export OPENAI_BASE_URL='http://your-api-url'  # 可选
```

**方式B：创建.env文件**
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "OPENAI_BASE_URL=http://your-api-url" >> .env  # 可选
```

### 3. 运行分析

```bash
# 使用ICLR 2025数据集
python run_iclr_analysis.py

# 或使用自定义数据
python main.py --input your_data.json --format json
```

### 4. 查看结果

```bash
# 分析结果保存在 results/ 目录
ls results/
# - analysis_report.txt      # 文本报告
# - analysis_results.json    # 详细结果
# - figures/                 # 可视化图表
```

## 数据格式

### 方式1：OpenReview JSON格式（推荐）

系统支持直接加载OpenReview导出的JSON文件：

```bash
ICLR_2025_CLEAN/
├── Paper_Title_1_paperid/
│   └── data.json
├── Paper_Title_2_paperid/
│   └── data.json
└── ...
```

每个`data.json`文件应包含OpenReview标准字段（title, abstract, reviews等）。

**使用方法：**
```python
pipeline.load_from_openreview("../ICLR_2025_CLEAN")
```

### 方式2：标准JSON格式

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

**使用方法：**
```python
pipeline.load_data("data/reviews.json", format="json")
```

### 方式3：CSV格式

CSV文件需要包含以下列：
- `paper_id`: 论文ID
- `title`: 论文标题
- `abstract`: 论文摘要
- `paper_content`: 论文全文（可选）
- `reviewer_id`: 审稿人ID
- `review_text`: 审稿意见
- `actual_score`: 实际打分

**使用方法：**
```python
pipeline.load_data("data/reviews.csv", format="csv")
```

## 详细使用方法

### 方式1：命令行工具（最简单）

```bash
# 使用ICLR 2025数据（完整流程，交互式）
python run_iclr_analysis.py

# 使用通用命令行工具
python main.py --input ../ICLR_2025_CLEAN --format openreview_json

# 分析JSON文件
python main.py --input data/reviews.json --format json

# 分析CSV文件
python main.py --input data/reviews.csv --format csv

# 指定输出目录
python main.py --input data.json --output my_results

# 只执行特定步骤
python main.py --input data.json --steps extract quantify

# 额外分析审稿相似度
python main.py --input data.json --similarity

# 禁用缓存
python main.py --input data.json --no-cache

# 清空缓存
python main.py --clear-cache

# 查看所有选项
python main.py --help
```

### 方式2：使用Python API（推荐）

```python
from pipeline import ReviewBiasAnalysisPipeline
from pathlib import Path

# 初始化pipeline（可指定输出目录）
pipeline = ReviewBiasAnalysisPipeline(
    output_dir=Path("./results")
)

# 加载OpenReview格式数据（推荐）
pipeline.load_from_openreview("../ICLR_2025_CLEAN")

# 或加载标准JSON/CSV数据
# pipeline.load_data("data/reviews.json", format="json")
# pipeline.load_data("data/reviews.csv", format="csv")

# 运行完整分析（自动执行所有步骤）
summary = pipeline.run_full_analysis()

# 保存结果
pipeline.save_results()
pipeline.generate_report()
```

### 方式3：分步执行（高级用户）

```python
from pipeline import ReviewBiasAnalysisPipeline

# 初始化
pipeline = ReviewBiasAnalysisPipeline()

# 1. 加载数据
pipeline.load_data("data/reviews.json", format="json")

# 2. 提取优缺点
pipeline.extract_pros_cons()

# 3. 量化优缺点权重
pipeline.quantify_weights()

# 4. 计算偏差
bias_results = pipeline.analyze_bias()

# 5. 生成可视化
pipeline.generate_visualizations()

# 6. 保存结果和报告
pipeline.save_results()
pipeline.generate_report()

# 7. 获取统计摘要
summary = pipeline.get_summary()
print(summary)
```

### 方式4：仅处理部分数据（测试用）

```python
pipeline = ReviewBiasAnalysisPipeline()
pipeline.load_from_openreview("../ICLR_2025_CLEAN")

# 只处理前2篇论文
pipeline.papers = pipeline.papers[:2]

# 运行分析
summary = pipeline.run_full_analysis()
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

## 输出说明

系统会在`results/`目录下生成以下文件：

### 文本报告
- `analysis_report.txt` - 分析报告摘要
- `analysis_results.json` - 完整的分析结果（JSON格式）
- `processed_papers.json` - 处理后的论文数据

### 可视化图表（figures/目录）
- `bias_distribution.png` - 偏差分布图（直方图+KDE）
- `score_comparison.png` - 期望分数vs实际分数散点图（支持大规模数据的透明度处理）
- `bias_by_paper.png` - 论文偏差分析图（大规模数据下自动切换为分布图+极端案例分析）
- `consistency_comparison.png` - 一致性对比图（大规模数据下自动切换为标准差相关性散点图）
- `bias_heatmap.png` - 偏差热力图（大规模数据下自动采样代表性案例）
- `score_boxplot.png` - 分数箱线图对比
- `category_analysis.png` - 优缺点类别统计图
- `bias_vs_actual_score.png` - 偏差与实际分数关系图
- `bias_by_reviewer_count.png` - 偏差与审稿人数量的关系图（新增）

### 日志文件（logs/目录）
- `analysis_YYYYMMDD_HHMMSS.log` - 运行日志

## 注意事项

1. **中文字体支持**：系统已配置Linux/Windows跨平台中文字体支持，图表标题和标签可正常显示中文
2. **API调用成本**：本系统会频繁调用LLM API，建议先用少量数据测试
3. **缓存机制**：系统默认启用缓存（`./cache`目录），避免重复调用API
4. **错误处理**：API调用失败会自动重试，最多3次
5. **数据量控制**：处理大量数据时建议分批进行，可先用`pipeline.papers[:N]`限制数量

## 常见问题

### Q1: 图表中文显示为乱码怎么办？
**A:** 系统已修复字体问题，支持Linux系统的中文显示。如仍有问题，请安装中文字体：
```bash
# Ubuntu/Debian
sudo apt install fonts-noto-cjk

# 清除matplotlib字体缓存
rm -rf ~/.cache/matplotlib
```

### Q2: 如何控制API调用成本？
**A:** 有以下几种方式：
- 先用少量数据测试：`pipeline.papers = pipeline.papers[:2]`
- 启用缓存（默认已启用）
- 使用本地模型或更便宜的API

### Q3: 如何查看分析进度？
**A:** 系统会实时输出进度信息到控制台和日志文件。

### Q4: 支持哪些数据格式？
**A:** 支持三种格式：
- OpenReview JSON（推荐，直接加载OpenReview导出数据）
- 标准JSON
- CSV

### Q5: 如何自定义输出目录？
**A:** 在初始化时指定：
```python
from pathlib import Path
pipeline = ReviewBiasAnalysisPipeline(output_dir=Path("./my_results"))
```

## 许可证

MIT License




