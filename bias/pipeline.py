"""
主Pipeline模块
整合所有模块，提供完整的审稿偏差分析流程
"""

from typing import List, Optional, Union
from pathlib import Path
import json

from config import Config
from data_loader import DataLoader, Paper
from feature_extractor import FeatureExtractor
from llm_quantifier import LLMQuantifier
from bias_analyzer import BiasAnalyzer, BiasAnalysisResult
from visualizer import Visualizer
from utils import logger


class ReviewBiasAnalysisPipeline:
    """审稿偏差分析Pipeline"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        output_dir: Optional[Path] = None
    ):
        """
        初始化Pipeline
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 模型名称
            output_dir: 输出目录
        """
        # 验证配置
        Config.validate()
        Config.display()
        
        # 初始化各模块
        self.data_loader = DataLoader()
        self.feature_extractor = FeatureExtractor(api_key, base_url, model)
        self.quantifier = LLMQuantifier(api_key, base_url, model)
        self.analyzer = BiasAnalyzer()
        self.visualizer = Visualizer(output_dir)
        
        # 数据存储
        self.papers: List[Paper] = []
        self.analysis_results: List[BiasAnalysisResult] = []
        
        logger.info("=" * 70)
        logger.info("审稿偏差分析Pipeline已初始化")
        logger.info("=" * 70)
    
    # ========== 数据加载 ==========
    
    def load_data(
        self,
        file_path: Union[str, Path],
        format: str = "json"
    ) -> 'ReviewBiasAnalysisPipeline':
        """
        加载数据
        
        Args:
            file_path: 数据文件路径
            format: 数据格式 ('json', 'csv', 'openreview_json')
            
        Returns:
            self（支持链式调用）
        """
        logger.info(f"\n{'='*70}")
        logger.info("步骤 1: 数据加载")
        logger.info(f"{'='*70}")
        
        if format == "json":
            self.papers = self.data_loader.load_from_json(file_path)
        elif format == "csv":
            self.papers = self.data_loader.load_from_csv(file_path)
        elif format == "openreview_json":
            self.papers = self.data_loader.load_from_openreview_json(file_path)
        else:
            raise ValueError(f"不支持的数据格式: {format}")
        
        self.data_loader.display_statistics()
        
        return self
    
    def load_from_openreview(
        self,
        directory: Union[str, Path]
    ) -> 'ReviewBiasAnalysisPipeline':
        """
        从OpenReview目录加载数据
        
        Args:
            directory: OpenReview数据目录
            
        Returns:
            self（支持链式调用）
        """
        return self.load_data(directory, format="openreview_json")
    
    # ========== 特征提取 ==========
    
    def load_paper_pdfs(self) -> 'ReviewBiasAnalysisPipeline':
        """
        为当前列表中的所有论文加载 PDF 内容（如果尚未加载）
        实现适配性优化：仅处理需要分析的论文
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"适配性优化：正在为 {len(self.papers)} 篇待分析论文提取 PDF 文本")
        logger.info(f"{'='*70}")
        
        for paper in self.papers:
            if not paper.paper_content and paper.source_dir:
                self.data_loader._load_pdf_content(paper, paper.source_dir)
        
        return self

    def extract_pros_cons(self, unified: bool = True) -> 'ReviewBiasAnalysisPipeline':
        """
        提取优缺点
        
        Args:
            unified: 是否使用统一提取模式
            
        Returns:
            self（支持链式调用）
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"步骤 2: 特征提取（{'统一整合模式' if unified else '独立提取模式'}）")
        logger.info(f"{'='*70}")
        
        if not self.papers:
            raise ValueError("请先加载数据")
        
        self.papers = self.feature_extractor.extract_from_papers(self.papers, unified=unified)
        
        if not unified:
            self.feature_extractor.display_extraction_summary(self.papers)
        
        return self
    
    # ========== 权重量化 ==========
    
    def quantify_weights(self, unified: bool = True) -> 'ReviewBiasAnalysisPipeline':
        """
        量化优缺点权重
        
        Args:
            unified: 是否使用统一量化模式
            
        Returns:
            self（支持链式调用）
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"步骤 3: 权重量化（{'统一量化模式' if unified else '独立量化模式'}）")
        logger.info(f"{'='*70}")
        
        if not self.papers:
            raise ValueError("请先加载数据")
        
        # 检查是否已提取优缺点
        has_extracted = False
        if unified:
            if any(p.unified_pros or p.unified_cons for p in self.papers):
                has_extracted = True
        else:
            if any(r.pros or r.cons for p in self.papers for r in p.reviews):
                has_extracted = True
                
        if not has_extracted:
            logger.warning("未检测到已提取的优缺点，将先执行特征提取步骤")
            self.extract_pros_cons(unified=unified)
        
        self.papers = self.quantifier.quantify_papers(self.papers, unified=unified)
        self.quantifier.display_quantification_summary(self.papers)
        
        return self
    
    # ========== 偏差分析 ==========
    
    def analyze_bias(self) -> List[BiasAnalysisResult]:
        """
        分析偏差
        
        Returns:
            偏差分析结果列表
        """
        logger.info(f"\n{'='*70}")
        logger.info("步骤 4: 偏差分析")
        logger.info(f"{'='*70}")
        
        if not self.papers:
            raise ValueError("请先加载数据")
        
        # 检查是否已量化权重
        if self.papers[0].reviews[0].expected_score is None:
            logger.warning("未检测到已量化的权重，将先执行权重量化步骤")
            self.quantify_weights()
        
        self.analysis_results = self.analyzer.analyze_papers(self.papers)
        self.analyzer.display_summary(self.analysis_results)
        
        return self.analysis_results
    
    # ========== 可视化 ==========
    
    def generate_visualizations(self) -> 'ReviewBiasAnalysisPipeline':
        """
        生成可视化图表
        
        Returns:
            self（支持链式调用）
        """
        logger.info(f"\n{'='*70}")
        logger.info("步骤 5: 生成可视化")
        logger.info(f"{'='*70}")
        
        if not self.analysis_results:
            logger.warning("未检测到分析结果，将先执行偏差分析步骤")
            self.analyze_bias()
        
        self.visualizer.generate_all_plots(self.papers, self.analysis_results)
        
        return self
    
    # ========== 完整流程 ==========
    
    def run_full_analysis(self, unified: bool = True) -> dict:
        """
        运行完整的分析流程
        
        Args:
            unified: 是否使用统一整合模式
            
        Returns:
            分析结果摘要字典
        """
        logger.info("\n" + "="*70)
        logger.info(f"开始完整分析流程 ({'统一整合模式' if unified else '独立模式'})")
        logger.info("="*70)
        
        # 0. 适配性优化：仅提取当前需要分析的论文 PDF 文本
        self.load_paper_pdfs()
        
        # 1. 特征提取
        self.extract_pros_cons(unified=unified)
        
        # 2. 权重量化
        self.quantify_weights(unified=unified)
        
        # 3. 保存每篇论文的详细报告
        self.save_individual_reports()
        
        # 4. 偏差分析
        self.analyze_bias()
        
        # 5. 生成可视化
        self.generate_visualizations()
        
        # 6. 生成摘要
        summary = {
            "data_statistics": self.data_loader.get_statistics(),
            "extraction_summary": self.feature_extractor.get_extraction_summary(self.papers),
            "quantification_summary": self.quantifier.get_quantification_summary(self.papers),
            "bias_statistics": self.analyzer.global_statistics(self.analysis_results),
        }
        
        logger.info("\n" + "="*70)
        logger.info("完整分析流程已完成！")
        logger.info("="*70)
        
        return summary
    
    # ========== 结果保存 ==========
    
    def save_individual_reports(self, output_dir: Optional[Union[str, Path]] = None):
        """
        为每篇论文保存详细的分析报告
        
        Args:
            output_dir: 输出目录，默认为 Config.DETAILS_DIR
        """
        logger.info(f"\n{'='*70}")
        logger.info("额外步骤: 保存每篇论文的详细分析报告")
        logger.info(f"{'='*70}")
        
        if not self.papers:
            logger.warning("没有论文数据，跳过详细报告保存")
            return
            
        details_dir = Path(output_dir) if output_dir else Config.DETAILS_DIR
        details_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"正在保存详细报告到: {details_dir}")
        
        for paper in self.papers:
            file_name = f"{paper.paper_id.replace('/', '_')}_details.md"
            file_path = details_dir / file_name
            
            content = self._generate_paper_report_content(paper)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        logger.info(f"已成功保存 {len(self.papers)} 篇论文的详细报告")

    def _generate_paper_report_content(self, paper: Paper) -> str:
        """生成单篇论文的报告内容"""
        content = [
            f"# 论文详细分析报告: {paper.title}",
            f"\n**论文ID:** {paper.paper_id}",
            f"\n## 摘要\n{paper.abstract}",
            f"\n## 审稿分析\n"
        ]
        
        for i, review in enumerate(paper.reviews):
            content.append(f"### 审稿人 {review.reviewer_id}")
            content.append(f"- **实际分数:** {review.actual_score}")
            content.append(f"- **量化期望分数:** {review.expected_score:.2f}" if review.expected_score is not None else "- **量化期望分数:** 未计算")
            content.append(f"- **偏差 (Actual - Expected):** {review.bias:+.2f}" if review.bias is not None else "- **偏差:** 未计算")
            
            content.append("\n#### 优点 (Pros)")
            if review.pros_weights:
                for pw in review.pros_weights:
                    content.append(f"- **[{pw.get('category', '未分类')}]** {pw.get('description', '')}")
                    content.append(f"  - 权重: `{pw.get('weight', 0):+.2f}`")
                    content.append(f"  - 理由: {pw.get('reasoning', '无')}")
            elif review.pros:
                for p in review.pros:
                    content.append(f"- **[{p.get('category', '未分类')}]** {p.get('description', '')}")
            else:
                content.append("- (无)")
                
            content.append("\n#### 缺点 (Cons)")
            if review.cons_weights:
                for cw in review.cons_weights:
                    content.append(f"- **[{cw.get('category', '未分类')}]** {cw.get('description', '')}")
                    content.append(f"  - 权重: `{cw.get('weight', 0):+.2f}`")
                    content.append(f"  - 理由: {cw.get('reasoning', '无')}")
            elif review.cons:
                for c in review.cons:
                    content.append(f"- **[{c.get('category', '未分类')}]** {c.get('description', '')}")
            else:
                content.append("- (无)")
            
            content.append("\n" + "-"*30 + "\n")
            
        return "\n".join(content)

    def save_results(self, output_file: Optional[Union[str, Path]] = None):
        """
        保存分析结果
        
        Args:
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = Config.OUTPUT_DIR / "analysis_results.json"
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备输出数据
        output_data = {
            "papers": [paper.to_dict() for paper in self.papers],
            "analysis_results": [result.to_dict() for result in self.analysis_results],
            "global_statistics": self.analyzer.global_statistics(self.analysis_results) if self.analysis_results else {},
        }
        
        # 保存为JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存到: {output_file}")
    
    def save_papers(self, output_file: Optional[Union[str, Path]] = None):
        """
        保存处理后的论文数据
        
        Args:
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = Config.OUTPUT_DIR / "processed_papers.json"
        else:
            output_file = Path(output_file)
        
        self.data_loader.save_to_json(output_file)
    
    def generate_report(self, output_file: Optional[Union[str, Path]] = None):
        """
        生成文本报告
        
        Args:
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = Config.OUTPUT_DIR / "analysis_report.txt"
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.analysis_results:
            logger.warning("没有分析结果，跳过报告生成")
            return
        
        report = self.analyzer.generate_summary_report(self.analysis_results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"分析报告已保存到: {output_file}")
    
    # ========== 高级功能 ==========
    
    def analyze_review_similarity(self) -> dict:
        """
        分析审稿相似度
        
        Returns:
            相似度分析结果
        """
        logger.info(f"\n{'='*70}")
        logger.info("额外分析: 审稿相似度")
        logger.info(f"{'='*70}")
        
        similarities = []
        
        for paper in self.papers:
            paper_similarities = self.feature_extractor.analyze_paper_review_similarity(paper)
            similarities.extend(paper_similarities)
        
        if not similarities:
            logger.warning("没有足够的审稿数据进行相似度分析")
            return {}
        
        # 统计分析
        import numpy as np
        
        overall_sims = [s['overall_similarity'] for s in similarities]
        score_diffs = [s['score_diff'] for s in similarities]
        
        # 计算相关性：相似度 vs 分数差异
        correlation = np.corrcoef(overall_sims, score_diffs)[0, 1]
        
        summary = {
            "total_comparisons": len(similarities),
            "avg_similarity": np.mean(overall_sims),
            "avg_score_diff": np.mean(score_diffs),
            "correlation": correlation,
            "interpretation": (
                "相似度越高，分数差异越小" if correlation < -0.3
                else "相似度越高，分数差异越大（反常）" if correlation > 0.3
                else "相似度与分数差异无明显关系"
            ),
            "details": similarities
        }
        
        logger.info(f"平均相似度: {summary['avg_similarity']:.3f}")
        logger.info(f"平均分数差异: {summary['avg_score_diff']:.2f}")
        logger.info(f"相关系数: {summary['correlation']:.3f}")
        logger.info(f"结论: {summary['interpretation']}")
        
        return summary
    
    def identify_problematic_papers(self, threshold: float = 2.0) -> List[dict]:
        """
        识别问题论文（高偏差）
        
        Args:
            threshold: 偏差阈值
            
        Returns:
            问题论文列表
        """
        if not self.analysis_results:
            logger.warning("请先运行偏差分析")
            return []
        
        problematic = self.analyzer.identify_high_bias_cases(
            self.analysis_results, 
            threshold=threshold
        )
        
        logger.info(f"识别出 {len(problematic)} 个高偏差案例")
        
        return problematic
    
    def get_summary(self) -> dict:
        """
        获取完整摘要
        
        Returns:
            摘要字典
        """
        if not self.analysis_results:
            logger.warning("请先运行分析")
            return {}
        
        return {
            "data": self.data_loader.get_statistics(),
            "extraction": self.feature_extractor.get_extraction_summary(self.papers),
            "quantification": self.quantifier.get_quantification_summary(self.papers),
            "bias_analysis": self.analyzer.global_statistics(self.analysis_results),
        }


if __name__ == "__main__":
    # 测试Pipeline
    import numpy as np
    from data_loader import Paper, Review
    
    # 创建测试数据
    test_papers = []
    for i in range(3):
        paper = Paper(
            paper_id=f"paper_{i}",
            title=f"Test Paper {i}: An Innovative Approach",
            abstract="This paper presents a novel method for solving complex problems."
        )
        
        for j in range(3):
            review = Review(
                reviewer_id=f"reviewer_{j}",
                review_text=f"""
                This paper has several strengths and weaknesses.
                
                Strengths:
                - Novel approach
                - Good experimental design
                
                Weaknesses:
                - Limited baseline comparisons
                - Writing could be improved
                """,
                actual_score=np.random.uniform(5, 9)
            )
            paper.add_review(review)
        
        test_papers.append(paper)
    
    # 保存测试数据
    test_data_file = Path("test_reviews.json")
    data_to_save = [p.to_dict() for p in test_papers]
    with open(test_data_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    print("\n✓ 测试数据已创建")
    print(f"数据文件: {test_data_file}")
    print("\n可以运行以下代码进行测试:")
    print("""
# 初始化pipeline
pipeline = ReviewBiasAnalysisPipeline()

# 加载数据
pipeline.load_data("test_reviews.json")

# 运行完整分析（需要配置API密钥）
# results = pipeline.run_full_analysis()

# 或分步执行
# pipeline.extract_pros_cons()
# pipeline.quantify_weights()
# pipeline.analyze_bias()
# pipeline.generate_visualizations()

# 保存结果
# pipeline.save_results()
# pipeline.generate_report()
    """)
    
    # 清理
    test_data_file.unlink()




