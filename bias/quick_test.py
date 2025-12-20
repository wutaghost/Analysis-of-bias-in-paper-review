#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 分析2篇论文
"""

from pipeline import ReviewBiasAnalysisPipeline

print("开始快速测试（只分析2篇论文）...\n")

# 初始化pipeline
pipeline = ReviewBiasAnalysisPipeline()

# 加载ICLR数据
pipeline.load_from_openreview('../ICLR_2025_CLEAN')

# 只处理前2篇论文（快速测试）
pipeline.papers = pipeline.papers[:2]
print(f"\n限制为前2篇论文进行测试\n")

# 运行完整分析
results = pipeline.run_full_analysis()

print("\n" + "="*70)
print("✅ 快速测试完成！")
print("="*70)
print(f"\n查看结果:")
print(f"  文本报告: cat results/analysis_report.txt")
print(f"  可视化图表: ls results/figures/")
print("\n如果测试成功，可以运行完整分析:")
print("  python run_iclr_analysis.py")
print("="*70)



