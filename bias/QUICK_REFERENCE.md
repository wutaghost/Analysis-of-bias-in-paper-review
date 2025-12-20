# 快速参考指南

## 安装和配置

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

**方式1: 环境变量**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**方式2: .env文件**
```bash
# 创建 .env 文件
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "MODEL_NAME=gpt-4o-mini" >> .env
```

### 3. 快速检查
```bash
python quick_start.py
```

## 基本使用

### 命令行方式

```bash
# 分析JSON数据
python main.py --input data/reviews.json

# 分析ICLR数据
python main.py --input ../ICLR_2025_CLEAN --format openreview_json

# 指定输出目录
python main.py --input data.json --output results_custom

# 只执行特定步骤
python main.py --input data.json --steps extract quantify analyze

# 包含相似度分析
python main.py --input data.json --similarity

# 自定义偏差阈值
python main.py --input data.json --bias-threshold 1.5
```

### Python API方式

```python
from pipeline import ReviewBiasAnalysisPipeline

# 初始化
pipeline = ReviewBiasAnalysisPipeline()

# 加载数据
pipeline.load_data("data/reviews.json")

# 运行完整分析
results = pipeline.run_full_analysis()

# 保存结果
pipeline.save_results()
pipeline.generate_report()
```

## 数据格式

### JSON格式

```json
[
  {
    "paper_id": "paper_001",
    "title": "论文标题",
    "abstract": "论文摘要",
    "reviews": [
      {
        "reviewer_id": "reviewer_1",
        "review_text": "审稿意见全文",
        "actual_score": 7.5
      }
    ]
  }
]
```

### CSV格式

必需列：
- `paper_id`: 论文ID
- `title`: 论文标题
- `abstract`: 论文摘要
- `reviewer_id`: 审稿人ID
- `review_text`: 审稿意见
- `actual_score`: 实际分数

## 分步执行

```python
from pipeline import ReviewBiasAnalysisPipeline

pipeline = ReviewBiasAnalysisPipeline()

# 步骤1: 加载数据
pipeline.load_data("data.json")

# 步骤2: 提取优缺点
pipeline.extract_pros_cons()

# 步骤3: 量化权重
pipeline.quantify_weights()

# 步骤4: 分析偏差
results = pipeline.analyze_bias()

# 步骤5: 生成可视化
pipeline.generate_visualizations()

# 保存结果
pipeline.save_results()
pipeline.generate_report()
```

## 高级功能

### 分析审稿相似度

```python
similarity = pipeline.analyze_review_similarity()
print(f"平均相似度: {similarity['avg_similarity']:.3f}")
```

### 识别高偏差案例

```python
problematic = pipeline.identify_problematic_papers(threshold=2.0)
print(f"发现 {len(problematic)} 个高偏差案例")
```

### 自定义分析

```python
# 访问原始数据
papers = pipeline.papers
results = pipeline.analysis_results

# 自定义统计
for paper in papers:
    for review in paper.reviews:
        print(f"论文: {paper.title}")
        print(f"审稿人: {review.reviewer_id}")
        print(f"实际分数: {review.actual_score}")
        print(f"期望分数: {review.expected_score}")
        print(f"偏差: {review.bias:+.2f}")
```

## 输出文件

分析完成后，结果保存在 `results/` 目录：

```
results/
├── figures/                        # 可视化图表
│   ├── bias_distribution.png      # 偏差分布
│   ├── score_comparison.png       # 分数对比
│   ├── bias_by_paper.png         # 论文偏差
│   ├── consistency_comparison.png # 一致性对比
│   ├── bias_heatmap.png          # 热力图
│   ├── score_boxplot.png         # 箱线图
│   ├── category_analysis.png     # 类别分析
│   └── bias_vs_actual_score.png  # 偏差-分数关系
│
├── logs/                          # 日志文件
├── analysis_results.json         # 完整结果
├── processed_papers.json         # 处理后的数据
├── analysis_report.txt           # 文本报告
├── high_bias_cases.json          # 高偏差案例
└── high_bias_cases.csv           # 高偏差案例(CSV)
```

## 常见问题

### Q1: API调用太慢怎么办？
A: 系统自动启用缓存，相同输入不会重复调用API。首次运行较慢是正常的。

### Q2: 如何减少API费用？
A:
- 使用 `gpt-4o-mini` 模型（默认）
- 先用小样本测试
- 确保缓存启用
- 使用 `--no-cache` 前先备份缓存

### Q3: 如何处理大量数据？
A:
- 分批处理
- 使用命令行指定部分数据
- 在代码中限制论文数量：`pipeline.papers = pipeline.papers[:N]`

### Q4: 图表中文显示乱码？
A: 确保系统安装了中文字体，或修改 `visualizer.py` 中的字体配置。

### Q5: 如何清空缓存？
A:
```bash
python main.py --clear-cache
```

或手动删除 `cache/` 目录。

### Q6: 如何使用自己的模型？
A:
```bash
python main.py --input data.json --model gpt-4
```

或在Python中：
```python
pipeline = ReviewBiasAnalysisPipeline(model='gpt-4')
```

## 性能优化建议

1. **首次运行**: 用少量数据测试（5-10篇论文）
2. **启用缓存**: 默认启用，避免重复调用
3. **批量处理**: 使用命令行工具处理大量数据
4. **并行处理**: 暂不支持，但可以通过分批运行实现
5. **监控成本**: 注意查看OpenAI使用情况

## 核心公式

### 期望分数
```
Expected_Score = BASE_SCORE + Σ(Pros_Weights) + Σ(Cons_Weights)
```

### 偏差
```
Bias = Actual_Score - Expected_Score
```

### 一致性比率
```
CV = std / mean
Consistency = 1 / (1 + CV)
Consistency_Ratio = Expected_Consistency / Actual_Consistency
```

## 统计指标说明

- **MAE (平均绝对误差)**: 偏差绝对值的平均
- **RMSE (均方根误差)**: 偏差平方的均方根
- **Pearson相关系数**: 期望分数与实际分数的线性相关
- **R²**: 决定系数，表示相关程度
- **p值**: 假设检验的显著性水平
- **一致性比率**: >1表示期望分数更一致

## 实际案例

### 案例1: 分析单个会议数据

```bash
# 1. 准备数据
cd /home/xingzhuang/workplace/yls/bias

# 2. 运行分析
python run_iclr_analysis.py

# 3. 查看结果
ls results/iclr_2025/
```

### 案例2: 比较多个会议

```python
from pipeline import ReviewBiasAnalysisPipeline
import pandas as pd

conferences = ['ICLR_2025', 'NeurIPS_2024']
results = {}

for conf in conferences:
    pipeline = ReviewBiasAnalysisPipeline(
        output_dir=f"results/{conf}"
    )
    pipeline.load_from_openreview(f"../{conf}")
    results[conf] = pipeline.run_full_analysis()

# 比较结果
comparison = pd.DataFrame([
    {
        '会议': conf,
        '平均偏差': r['bias_statistics']['bias_statistics']['mean'],
        'MAE': r['bias_statistics']['bias_statistics']['mae'],
    }
    for conf, r in results.items()
])
print(comparison)
```

### 案例3: 导出到Excel

```python
import pandas as pd

pipeline = ReviewBiasAnalysisPipeline()
pipeline.load_data("data.json")
pipeline.run_full_analysis()

# 导出详细数据
data = []
for paper in pipeline.papers:
    for review in paper.reviews:
        data.append({
            '论文标题': paper.title,
            '审稿人': review.reviewer_id,
            '实际分数': review.actual_score,
            '期望分数': review.expected_score,
            '偏差': review.bias,
            '优点数': len(review.pros),
            '缺点数': len(review.cons),
        })

df = pd.DataFrame(data)
df.to_excel('results/detailed_analysis.xlsx', index=False)
```

## 获取帮助

```bash
# 查看命令行帮助
python main.py --help

# 运行示例
python example_usage.py

# 系统检查
python quick_start.py

# 查看文档
cat README.md
cat PROJECT_STRUCTURE.md
```

## 联系和反馈

- 查看日志文件: `results/logs/`
- 阅读完整文档: `README.md`
- 查看项目结构: `PROJECT_STRUCTURE.md`
- 运行测试脚本确认系统正常




