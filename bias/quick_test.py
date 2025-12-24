#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 仅测试2篇论文

用于验证系统功能是否正常工作，避免长时间等待和高API费用。

新流程（四步骤）:
1. 特征提取: 使用LLM独立提取每个审稿人的优缺点
2. 匿名化处理: 去除审稿人信息，随机打乱顺序（代码逻辑）
3. 权重量化: 基于匿名优缺点+论文全文，LLM量化权重
4. 匹配计算: 代码逻辑匹配回审稿人，线性相加计算分数
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ReviewBiasAnalysisPipeline
from config import Config
from utils import logger


def main():
    """快速测试主函数"""
    
    print("="*70)
    print("🧪 快速测试脚本 - 仅分析2篇论文")
    print("="*70)
    
    # ICLR 2025数据目录
    iclr_data_dir = Path(__file__).parent.parent / "ICLR_2025_CLEAN"
    
    # 检查数据目录
    if not iclr_data_dir.exists():
        print(f"\n❌ 错误: 数据目录不存在: {iclr_data_dir}")
        print("请确保ICLR_2025_CLEAN目录在正确的位置")
        return 1
    
    print(f"\n📂 数据目录: {iclr_data_dir}")
    
    # 检查配置
    try:
        Config.validate()
        print("✓ 配置验证成功")
    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
        print("\n请先配置OpenAI API密钥：")
        print("  export OPENAI_API_KEY='your-key'")
        print("  或创建 .env 文件")
        return 1
    
    print("\n" + "-"*70)
    print("四步骤流程说明:")
    print("  步骤1: 特征提取 - LLM独立提取每个审稿人的优缺点")
    print("  步骤2: 匿名化处理 - 代码逻辑去除审稿人信息+随机打乱")
    print("  步骤3: 权重量化 - LLM基于匿名优缺点+论文全文量化权重")
    print("  步骤4: 匹配计算 - 代码逻辑匹配回审稿人+线性相加")
    print("-"*70)
    
    try:
        # 初始化Pipeline
        print("\n📊 初始化分析Pipeline...")
        pipeline = ReviewBiasAnalysisPipeline(
            output_dir=Path("./results/quick_test")
        )
        
        # 加载数据
        print("\n📥 加载ICLR 2025数据...")
        pipeline.load_from_openreview(iclr_data_dir)
        
        total_papers = len(pipeline.papers)
        print(f"   找到 {total_papers} 篇论文")
        
        # 限制为2篇论文
        pipeline.papers = pipeline.papers[:2]
        print(f"   🎯 限制为前2篇论文进行测试")
        
        # 显示测试论文信息
        print("\n📝 测试论文:")
        for i, paper in enumerate(pipeline.papers):
            print(f"   {i+1}. {paper.title[:60]}...")
            print(f"      审稿人数: {len(paper.reviews)}")
        
        # 运行完整分析
        print("\n" + "="*70)
        print("🚀 开始四步骤分析...")
        print("="*70)
        
        results = pipeline.run_full_analysis()
        
        # 显示关键结果
        print("\n" + "="*70)
        print("📊 测试结果摘要")
        print("="*70)
        
        # 偏差统计
        bias_stats = results.get('bias_statistics', {}).get('bias_statistics', {})
        if bias_stats:
            print(f"\n偏差统计:")
            print(f"  平均偏差: {bias_stats.get('mean', 0):+.3f}")
            print(f"  偏差标准差: {bias_stats.get('std', 0):.3f}")
        
        # 中间文件
        print(f"\n📁 中间文件输出:")
        output_files = results.get('output_files', {})
        for key, path in output_files.items():
            print(f"  {key}: {path}")
        
        # 每个审稿人的结果
        print(f"\n📋 各审稿人分析结果:")
        for paper in pipeline.papers:
            print(f"\n  论文: {paper.title[:50]}...")
            for review in paper.reviews:
                if review.expected_score is not None:
                    print(f"    审稿人 {review.reviewer_id}:")
                    print(f"      实际分数: {review.actual_score:.1f}")
                    print(f"      期望分数: {review.expected_score:.2f}")
                    print(f"      偏差: {review.bias:+.2f}")
                    print(f"      优点数: {len(review.pros_weights)}, 缺点数: {len(review.cons_weights)}")
        
        print("\n" + "="*70)
        print("✅ 快速测试完成！")
        print("="*70)
        
        print(f"\n📂 详细结果保存在: ./results/quick_test/")
        print(f"   - 每篇论文详情: paper_details/")
        print(f"   - 提取结果: extraction/")
        print(f"   - 匿名化数据: anonymized/")
        print(f"   - 量化结果: quantified/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
        return 130
    
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        logger.error("测试失败", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

