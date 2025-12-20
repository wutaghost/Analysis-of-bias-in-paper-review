# 快速入门指南

欢迎使用论文审稿偏差分析系统！本指南将帮助您在5分钟内开始使用。

## 🚀 快速开始（3步）

### 步骤1: 安装依赖
```bash
cd /home/xingzhuang/workplace/yls/bias
pip install -r requirements.txt
```

### 步骤2: 配置API密钥
```bash
# 创建.env文件并添加您的API密钥
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

### 步骤3: 运行快速检查
```bash
python quick_start.py
```

如果看到 `✓ 系统检查完成！`，说明安装成功！

## 📊 开始分析ICLR数据

### 方式1: 使用专用脚本（推荐）
```bash
python run_iclr_analysis.py
```

这个脚本会：
- 自动加载ICLR_2025_CLEAN目录的数据
- 运行完整的偏差分析
- 生成所有可视化图表
- 保存详细报告

### 方式2: 使用命令行工具
```bash
# 完整分析
python main.py --input ../ICLR_2025_CLEAN --format openreview_json

# 包含相似度分析
python main.py --input ../ICLR_2025_CLEAN --format openreview_json --similarity

# 指定输出目录
python main.py --input ../ICLR_2025_CLEAN --format openreview_json --output results/my_analysis
```

### 方式3: 使用Python API
```python
from pipeline import ReviewBiasAnalysisPipeline

# 初始化
pipeline = ReviewBiasAnalysisPipeline()

# 加载ICLR数据
pipeline.load_from_openreview("../ICLR_2025_CLEAN")

# 运行分析
results = pipeline.run_full_analysis()

# 查看结果
print("平均偏差:", results['bias_statistics']['bias_statistics']['mean'])

# 保存结果
pipeline.save_results()
pipeline.generate_report()
```

## 📁 查看结果

分析完成后，查看 `results/` 目录：

```bash
# 查看文本报告
cat results/analysis_report.txt

# 查看图表
ls results/figures/

# 查看详细数据
cat results/analysis_results.json
```

## 🎯 核心功能

### 1. 提取优缺点
系统使用LLM从审稿文本中自动提取：
- 优点列表（包含描述和类别）
- 缺点列表（包含描述和类别）

### 2. 量化权重
为每个优缺点赋予量化权重：
- 优点：正权重（+0.1 到 +2.0）
- 缺点：负权重（-0.1 到 -2.0）

### 3. 计算期望分数
```
期望分数 = 基准分数(5.0) + 所有优点权重之和 + 所有缺点权重之和
```

### 4. 分析偏差
```
偏差 = 实际分数 - 期望分数
```
- 正偏差：审稿人给分偏高
- 负偏差：审稿人给分偏低
- 零偏差：给分合理

### 5. 生成可视化
8种统计图表，全面展示偏差情况

## 📖 详细文档

- **README.md**: 项目完整说明
- **QUICK_REFERENCE.md**: 快速参考手册
- **PROJECT_STRUCTURE.md**: 项目结构和技术细节
- **example_usage.py**: 代码使用示例

## 🔧 常见场景

### 场景1: 只想看看结果，不想等太久
```bash
# 只分析前10篇论文（测试用）
python -c "
from pipeline import ReviewBiasAnalysisPipeline
pipeline = ReviewBiasAnalysisPipeline()
pipeline.load_from_openreview('../ICLR_2025_CLEAN')
pipeline.papers = pipeline.papers[:10]  # 限制为10篇
pipeline.run_full_analysis()
pipeline.save_results()
"
```

### 场景2: 已有审稿数据，想分析偏差
1. 准备JSON或CSV格式数据（参考README.md中的格式说明）
2. 运行：`python main.py --input your_data.json`

### 场景3: 想对比不同会议的偏差
```python
conferences = ['ICLR_2025', 'NeurIPS_2024', 'ICML_2024']

for conf in conferences:
    pipeline = ReviewBiasAnalysisPipeline(output_dir=f"results/{conf}")
    pipeline.load_from_openreview(f"../{conf}_CLEAN")
    pipeline.run_full_analysis()
```

### 场景4: 导出到Excel进行进一步分析
```python
import pandas as pd
from pipeline import ReviewBiasAnalysisPipeline

pipeline = ReviewBiasAnalysisPipeline()
pipeline.load_from_openreview("../ICLR_2025_CLEAN")
pipeline.run_full_analysis()

# 创建DataFrame
data = []
for paper in pipeline.papers:
    for review in paper.reviews:
        data.append({
            '论文': paper.title,
            '审稿人': review.reviewer_id,
            '实际分数': review.actual_score,
            '期望分数': review.expected_score,
            '偏差': review.bias,
        })

df = pd.DataFrame(data)
df.to_excel('results/analysis.xlsx', index=False)
print("已导出到 results/analysis.xlsx")
```

## ⚠️ 重要提示

### 关于API费用
- 本系统会调用OpenAI API，会产生费用
- 使用 `gpt-4o-mini` 模型成本较低（默认）
- 系统自动启用缓存，避免重复调用
- 建议先用小样本测试（5-10篇论文）

### 关于处理时间
- 提取优缺点：约5-10秒/审稿
- 量化权重：约5-10秒/审稿
- 总时间 = 审稿数量 × 10-20秒
- 例如：100条审稿约需15-30分钟

### 关于数据隐私
- 所有数据在本地处理
- LLM调用通过OpenAI API
- 注意不要上传敏感数据

## 🐛 遇到问题？

### 问题1: ImportError: No module named 'xxx'
```bash
pip install -r requirements.txt
```

### 问题2: API调用失败
- 检查API密钥是否正确
- 检查网络连接
- 查看 `results/logs/` 中的错误日志

### 问题3: 内存不足
- 减少处理的论文数量
- 分批处理数据

### 问题4: 中文乱码
- 确保系统安装了中文字体
- 或修改 `visualizer.py` 中的字体配置

### 问题5: 结果不理想
- 检查输入数据格式是否正确
- 查看提取的优缺点是否合理
- 可能需要调整Prompt模板（在`config.py`中）

## 💡 技巧和建议

1. **首次使用**: 用5-10篇论文测试，确认流程正常
2. **节省费用**: 确保缓存启用（默认启用）
3. **加速分析**: 第二次运行相同数据会很快（使用缓存）
4. **清空缓存**: `python main.py --clear-cache`
5. **监控进度**: 系统会实时显示进度和ETA

## 📞 获取帮助

```bash
# 查看命令行帮助
python main.py --help

# 运行测试
python quick_start.py

# 查看示例
python example_usage.py

# 查看文档
cat README.md
```

## 🎓 理解结果

### 关键指标解释

1. **平均偏差**: 
   - 接近0：审稿人给分较为合理
   - 显著偏离0：存在系统性偏差

2. **偏差标准差**:
   - 越大：不同审稿人之间差异越大
   - 越小：审稿相对一致

3. **一致性比率**:
   - \>1：期望分数比实际分数更一致（支持偏差假设）
   - <1：实际分数比期望分数更一致

4. **Pearson相关系数**:
   - 接近1：期望分数与实际分数高度正相关
   - 接近0：两者关系不大

## 下一步

- ✅ 运行 `python quick_start.py` 检查系统
- ✅ 运行 `python run_iclr_analysis.py` 分析ICLR数据
- ✅ 查看 `results/` 目录中的结果
- ✅ 阅读生成的报告和图表
- ✅ 根据需要调整分析参数

祝您使用愉快！🎉




