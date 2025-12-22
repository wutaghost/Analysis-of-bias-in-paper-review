"""
主程序入口
提供命令行接口运行审稿偏差分析
"""

import argparse
import sys
from pathlib import Path

from pipeline import ReviewBiasAnalysisPipeline
from config import Config
from utils import logger, cache_manager


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="审稿偏差分析系统 - 分析论文审稿中的偏差现象",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 从JSON文件分析
  python main.py --input data/reviews.json --format json
  
  # 2. 从OpenReview目录批量分析
  python main.py --input ../ICLR_2025_CLEAN --format openreview_json
  
  # 3. 从CSV文件分析
  python main.py --input data/reviews.csv --format csv
  
  # 4. 指定输出目录
  python main.py --input data/reviews.json --output results_custom
  
  # 5. 只执行特定步骤
  python main.py --input data/reviews.json --steps extract quantify
  
  # 6. 清空缓存
  python main.py --clear-cache
        """
    )
    
    # 输入参数
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="输入数据文件或目录路径"
    )
    
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["json", "csv", "openreview_json"],
        default="json",
        help="输入数据格式 (默认: json)"
    )
    
    # 输出参数
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出目录路径 (默认: ./results)"
    )
    
    # 执行步骤
    parser.add_argument(
        "-s", "--steps",
        type=str,
        nargs="+",
        choices=["extract", "quantify", "analyze", "visualize", "all"],
        default=["all"],
        help="要执行的步骤 (默认: all)"
    )
    
    # API配置
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API密钥 (优先级高于环境变量)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"使用的模型名称 (默认: {Config.MODEL_NAME})"
    )
    
    # 其他选项
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="清空缓存后退出"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用缓存"
    )
    
    parser.add_argument(
        "--similarity",
        action="store_true",
        help="额外分析审稿相似度"
    )
    
    parser.add_argument(
        "--bias-threshold",
        type=float,
        default=2.0,
        help="识别高偏差案例的阈值 (默认: 2.0)"
    )
    
    args = parser.parse_args()
    
    # 处理清空缓存
    if args.clear_cache:
        cache_manager.clear()
        print("✓ 缓存已清空")
        return 0
    
    # 检查输入
    if not args.input:
        parser.print_help()
        print("\n错误: 必须指定输入文件或目录 (--input)")
        return 1
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n错误: 输入路径不存在: {input_path}")
        return 1
    
    # 禁用缓存
    if args.no_cache:
        Config.ENABLE_CACHE = False
    
    try:
        # 初始化pipeline
        output_dir = Path(args.output) if args.output else None
        pipeline = ReviewBiasAnalysisPipeline(
            api_key=args.api_key,
            model=args.model,
            output_dir=output_dir
        )
        
        # 加载数据
        logger.info(f"从 {input_path} 加载数据...")
        pipeline.load_data(input_path, format=args.format)
        
        # 确定要执行的步骤
        steps = set(args.steps)
        if "all" in steps:
            steps = {"extract", "quantify", "analyze", "visualize"}
        
        # 执行步骤
        if "extract" in steps:
            pipeline.extract_pros_cons()
        
        if "quantify" in steps:
            pipeline.quantify_weights()
            # 保存每篇论文的详细报告
            pipeline.save_individual_reports()
        
        if "analyze" in steps:
            pipeline.analyze_bias()
        
        if "visualize" in steps:
            pipeline.generate_visualizations()
        
        # 额外分析：审稿相似度
        if args.similarity:
            similarity_results = pipeline.analyze_review_similarity()
            
            # 保存相似度结果
            import json
            similarity_file = Config.OUTPUT_DIR / "similarity_analysis.json"
            with open(similarity_file, 'w', encoding='utf-8') as f:
                json.dump(similarity_results, f, ensure_ascii=False, indent=2)
            logger.info(f"相似度分析结果已保存到: {similarity_file}")
        
        # 识别高偏差案例
        if "analyze" in steps:
            problematic = pipeline.identify_problematic_papers(
                threshold=args.bias_threshold
            )
            
            # 保存高偏差案例
            if problematic:
                import json
                import pandas as pd
                
                problematic_file = Config.OUTPUT_DIR / "high_bias_cases.json"
                with open(problematic_file, 'w', encoding='utf-8') as f:
                    json.dump(problematic, f, ensure_ascii=False, indent=2)
                
                # 同时保存为CSV
                df = pd.DataFrame(problematic)
                csv_file = Config.OUTPUT_DIR / "high_bias_cases.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                
                logger.info(f"高偏差案例已保存到: {problematic_file}")
        
        # 保存结果
        pipeline.save_results()
        pipeline.save_papers()
        pipeline.generate_report()
        
        # 最终总结
        logger.info("\n" + "="*70)
        logger.info("分析完成！")
        logger.info("="*70)
        logger.info(f"所有结果已保存到: {Config.OUTPUT_DIR}")
        logger.info("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n用户中断执行")
        return 130
    
    except Exception as e:
        logger.error(f"\n执行出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())




