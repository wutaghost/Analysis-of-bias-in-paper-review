"""
使用示例脚本
演示如何使用审稿偏差分析系统
"""

import json
from pathlib import Path

from pipeline import ReviewBiasAnalysisPipeline
from data_loader import Paper, Review


def example_1_basic_usage():
    """示例1: 基本使用流程"""
    print("\n" + "="*70)
    print("示例1: 基本使用流程")
    print("="*70)
    
    # 创建示例数据
    sample_data = [
        {
            "paper_id": "paper_001",
            "title": "Attention Is All You Need",
            "abstract": "We propose a new architecture based solely on attention mechanisms...",
            "reviews": [
                {
                    "reviewer_id": "reviewer_1",
                    "review_text": """
                    Summary: This paper presents the Transformer architecture.
                    
                    Strengths:
                    - Novel and elegant architecture
                    - Strong experimental results on machine translation
                    - Well-written and clear presentation
                    
                    Weaknesses:
                    - Limited analysis of computational complexity
                    - Missing comparisons with some recent methods
                    """,
                    "actual_score": 8
                },
                {
                    "reviewer_id": "reviewer_2",
                    "review_text": """
                    Summary: Introduces Transformer model for NLP tasks.
                    
                    Strengths:
                    - Innovative approach
                    - Comprehensive experiments
                    
                    Weaknesses:
                    - Computational cost not well addressed
                    - Some baselines missing
                    - Writing could be more concise
                    """,
                    "actual_score": 6
                }
            ]
        }
    ]
    
    # 保存示例数据
    data_file = Path("example_data.json")
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 示例数据已创建: {data_file}")
    print("\n运行分析需要配置OpenAI API密钥")
    print("请设置环境变量 OPENAI_API_KEY 或在 .env 文件中配置")
    print("\n分析代码示例:")
    print("""
    # 1. 初始化pipeline
    pipeline = ReviewBiasAnalysisPipeline()
    
    # 2. 加载数据
    pipeline.load_data("example_data.json")
    
    # 3. 运行完整分析
    results = pipeline.run_full_analysis()
    
    # 4. 保存结果
    pipeline.save_results()
    pipeline.generate_report()
    """)


def example_2_step_by_step():
    """示例2: 分步执行"""
    print("\n" + "="*70)
    print("示例2: 分步执行流程")
    print("="*70)
    
    print("""
# 初始化
pipeline = ReviewBiasAnalysisPipeline()

# 步骤1: 加载数据
pipeline.load_data("data/reviews.json", format="json")

# 步骤2: 提取优缺点
pipeline.extract_pros_cons()

# 步骤3: 量化权重
pipeline.quantify_weights()

# 步骤4: 分析偏差
results = pipeline.analyze_bias()

# 步骤5: 生成可视化
pipeline.generate_visualizations()

# 查看结果
summary = pipeline.get_summary()
print(summary)
    """)


def example_3_openreview_data():
    """示例3: 使用OpenReview数据"""
    print("\n" + "="*70)
    print("示例3: 使用OpenReview数据")
    print("="*70)
    
    print("""
# 从OpenReview目录加载数据
pipeline = ReviewBiasAnalysisPipeline()
pipeline.load_from_openreview("../ICLR_2025_CLEAN")

# 运行完整分析
results = pipeline.run_full_analysis()

# 分析审稿相似度
similarity = pipeline.analyze_review_similarity()

# 识别高偏差案例
problematic = pipeline.identify_problematic_papers(threshold=2.0)

print(f"发现 {len(problematic)} 个高偏差案例")
    """)


def example_4_custom_analysis():
    """示例4: 自定义分析"""
    print("\n" + "="*70)
    print("示例4: 自定义分析")
    print("="*70)
    
    print("""
from pipeline import ReviewBiasAnalysisPipeline
import pandas as pd

# 初始化
pipeline = ReviewBiasAnalysisPipeline()
pipeline.load_data("data/reviews.json")
pipeline.run_full_analysis()

# 自定义分析1: 找出偏差最大的论文
results = pipeline.analysis_results
max_bias_paper = max(results, key=lambda x: abs(x.bias_mean))
print(f"最大偏差论文: {max_bias_paper.paper_title}")
print(f"平均偏差: {max_bias_paper.bias_mean:.2f}")

# 自定义分析2: 统计各类别的权重分布
category_weights = {}
for paper in pipeline.papers:
    for review in paper.reviews:
        for pro in review.pros_weights:
            cat = pro.get('category', '其他')
            if cat not in category_weights:
                category_weights[cat] = []
            category_weights[cat].append(pro.get('weight', 0))

for cat, weights in category_weights.items():
    avg_weight = sum(weights) / len(weights)
    print(f"{cat}: 平均权重 = {avg_weight:.3f}")

# 自定义分析3: 导出详细数据到Excel
data = []
for paper in pipeline.papers:
    for review in paper.reviews:
        data.append({
            '论文ID': paper.paper_id,
            '论文标题': paper.title,
            '审稿人': review.reviewer_id,
            '实际分数': review.actual_score,
            '期望分数': review.expected_score,
            '偏差': review.bias,
            '优点数': len(review.pros),
            '缺点数': len(review.cons),
        })

df = pd.DataFrame(data)
df.to_excel('detailed_results.xlsx', index=False)
    """)


def example_5_api_configuration():
    """示例5: API配置"""
    print("\n" + "="*70)
    print("示例5: 自定义API配置")
    print("="*70)
    
    print("""
# 方式1: 通过环境变量
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key'
os.environ['OPENAI_BASE_URL'] = 'https://api.openai.com/v1'  # 可选
os.environ['MODEL_NAME'] = 'gpt-4o-mini'

pipeline = ReviewBiasAnalysisPipeline()

# 方式2: 通过构造函数参数
pipeline = ReviewBiasAnalysisPipeline(
    api_key='your-api-key',
    base_url='https://api.openai.com/v1',  # 可选
    model='gpt-4o-mini'
)

# 方式3: 使用.env文件
# 创建 .env 文件并添加:
# OPENAI_API_KEY=your-api-key
# OPENAI_BASE_URL=https://api.openai.com/v1
# MODEL_NAME=gpt-4o-mini

pipeline = ReviewBiasAnalysisPipeline()
    """)


def example_6_batch_processing():
    """示例6: 批量处理"""
    print("\n" + "="*70)
    print("示例6: 批量处理多个数据集")
    print("="*70)
    
    print("""
from pathlib import Path
from pipeline import ReviewBiasAnalysisPipeline

# 批量处理多个会议的数据
conferences = ['ICLR_2025', 'NeurIPS_2024', 'ICML_2024']

all_results = {}

for conf in conferences:
    print(f"\\n处理 {conf}...")
    
    # 为每个会议创建独立的输出目录
    pipeline = ReviewBiasAnalysisPipeline(
        output_dir=Path(f"results/{conf}")
    )
    
    # 加载数据
    pipeline.load_from_openreview(f"data/{conf}")
    
    # 运行分析
    results = pipeline.run_full_analysis()
    
    # 保存结果
    pipeline.save_results()
    pipeline.generate_report()
    
    all_results[conf] = results

# 比较不同会议的偏差情况
import pandas as pd

comparison = []
for conf, results in all_results.items():
    stats = results['bias_statistics']
    comparison.append({
        '会议': conf,
        '平均偏差': stats['bias_statistics']['mean'],
        '偏差标准差': stats['bias_statistics']['std'],
        'MAE': stats['bias_statistics']['mae'],
    })

df = pd.DataFrame(comparison)
print("\\n各会议偏差对比:")
print(df)
    """)


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("审稿偏差分析系统 - 使用示例")
    print("="*70)
    
    example_1_basic_usage()
    example_2_step_by_step()
    example_3_openreview_data()
    example_4_custom_analysis()
    example_5_api_configuration()
    example_6_batch_processing()
    
    print("\n" + "="*70)
    print("更多信息请参考 README.md")
    print("="*70)


if __name__ == "__main__":
    main()




