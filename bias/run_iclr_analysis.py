"""
运行ICLR 2025数据分析的示例脚本
"""

import sys
from pathlib import Path

from pipeline import ReviewBiasAnalysisPipeline
from config import Config
from utils import logger


def main():
    """主函数"""
    
    # ICLR 2025数据目录
    iclr_data_dir = Path("../ICLR_2025_CLEAN")
    
    # 检查数据目录是否存在
    if not iclr_data_dir.exists():
        print(f"错误: 数据目录不存在: {iclr_data_dir}")
        print("\n请确保ICLR_2025_CLEAN目录在正确的位置")
        print("当前脚本目录:", Path.cwd())
        return 1
    
    print("="*70)
    print("ICLR 2025 审稿偏差分析")
    print("="*70)
    print(f"\n数据目录: {iclr_data_dir.resolve()}")
    
    # 检查API密钥
    try:
        Config.validate()
    except ValueError as e:
        print(f"\n错误: {e}")
        print("\n请先配置OpenAI API密钥：")
        print("1. 设置环境变量: export OPENAI_API_KEY='your-key'")
        print("2. 或创建 .env 文件")
        return 1
    
    # 用户确认
    print(f"\n注意: 此分析将调用OpenAI API，可能产生费用。")
    print(f"数据目录包含多篇论文和审稿记录。")
    
    response = input("\n是否继续？ [y/N]: ")
    if response.lower() != 'y':
        print("已取消")
        return 0
    
    try:
        # 初始化Pipeline
        print("\n" + "="*70)
        print("初始化分析Pipeline...")
        print("="*70)
        
        pipeline = ReviewBiasAnalysisPipeline(
            output_dir=Path("./results/iclr_2025")
        )
        
        # 加载ICLR数据
        print("\n" + "="*70)
        print("加载ICLR 2025数据...")
        print("="*70)
        
        pipeline.load_from_openreview(iclr_data_dir)
        
        # 显示数据统计
        stats = pipeline.data_loader.get_statistics()
        print(f"\n成功加载:")
        print(f"  - 论文数: {stats['total_papers']}")
        print(f"  - 审稿记录数: {stats['total_reviews']}")
        print(f"  - 平均每篇论文的审稿数: {stats['avg_reviews_per_paper']:.1f}")
        
        # 询问是否只处理部分数据（用于测试）
        if stats['total_papers'] > 10:
            response = input(f"\n数据量较大({stats['total_papers']}篇)，是否只处理前10篇进行测试？ [y/N]: ")
            if response.lower() == 'y':
                pipeline.papers = pipeline.papers[:10]
                print(f"已限制为前10篇论文")
        
        # 运行完整分析
        print("\n" + "="*70)
        print("开始运行完整分析...")
        print("="*70)
        print("\n这可能需要一些时间，请耐心等待...")
        print("进度信息将实时显示在下方\n")
        
        results = pipeline.run_full_analysis()
        
        # 额外分析：审稿相似度
        print("\n" + "="*70)
        print("进行额外分析：审稿相似度...")
        print("="*70)
        
        similarity_results = pipeline.analyze_review_similarity()
        
        # 识别高偏差案例
        print("\n" + "="*70)
        print("识别高偏差案例...")
        print("="*70)
        
        problematic = pipeline.identify_problematic_papers(threshold=2.0)
        
        # 保存所有结果
        print("\n" + "="*70)
        print("保存结果...")
        print("="*70)
        
        pipeline.save_results()
        pipeline.save_papers()
        pipeline.generate_report()
        
        # 保存额外结果
        import json
        
        # 保存相似度结果
        similarity_file = Config.OUTPUT_DIR / "iclr_similarity_analysis.json"
        with open(similarity_file, 'w', encoding='utf-8') as f:
            json.dump(similarity_results, f, ensure_ascii=False, indent=2)
        logger.info(f"相似度分析结果已保存: {similarity_file}")
        
        # 保存高偏差案例
        if problematic:
            import pandas as pd
            
            problematic_file = Config.OUTPUT_DIR / "iclr_high_bias_cases.json"
            with open(problematic_file, 'w', encoding='utf-8') as f:
                json.dump(problematic, f, ensure_ascii=False, indent=2)
            
            df = pd.DataFrame(problematic)
            csv_file = Config.OUTPUT_DIR / "iclr_high_bias_cases.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"高偏差案例已保存: {problematic_file}")
        
        # 输出关键结果摘要
        print("\n" + "="*70)
        print("分析完成！关键结果摘要")
        print("="*70)
        
        bias_stats = results['bias_statistics']['bias_statistics']
        print(f"\n偏差统计:")
        print(f"  平均偏差: {bias_stats['mean']:+.3f}")
        print(f"  偏差标准差: {bias_stats['std']:.3f}")
        print(f"  MAE: {bias_stats['mae']:.3f}")
        print(f"  RMSE: {bias_stats['rmse']:.3f}")
        
        corr = results['bias_statistics']['correlation']
        print(f"\n相关性分析:")
        print(f"  Pearson相关系数: {corr['pearson_r']:.3f}")
        print(f"  R²: {corr['r_squared']:.3f}")
        
        t_test = results['bias_statistics']['hypothesis_tests']['paired_t_test']
        print(f"\n假设检验:")
        print(f"  {t_test['interpretation']}")
        print(f"  p值: {t_test['p_value']:.4f}")
        
        if similarity_results:
            print(f"\n相似度分析:")
            print(f"  平均相似度: {similarity_results['avg_similarity']:.3f}")
            print(f"  平均分数差异: {similarity_results['avg_score_diff']:.2f}")
            print(f"  {similarity_results['interpretation']}")
        
        print(f"\n高偏差案例: {len(problematic)} 个")
        
        print(f"\n所有结果已保存到: {Config.OUTPUT_DIR}")
        print(f"  - 分析报告: analysis_report.txt")
        print(f"  - 详细结果: analysis_results.json")
        print(f"  - 可视化图表: figures/")
        print(f"  - 高偏差案例: iclr_high_bias_cases.csv")
        
        print("\n" + "="*70)
        print("✓ 分析完成！")
        print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
        return 130
    
    except Exception as e:
        print(f"\n\n错误: {e}")
        logger.error("分析失败", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())




