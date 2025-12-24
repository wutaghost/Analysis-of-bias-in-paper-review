"""
主Pipeline模块
整合所有模块，提供完整的审稿偏差分析流程

新流程（四步骤）:
1. 特征提取: 使用LLM提取每个审稿人的优缺点 -> 输出文件
2. 匿名化处理: 去除审稿人信息，打乱顺序 -> 输出新文件
3. 权重量化: 基于匿名化文件+PDF内容，使用LLM量化 -> 输出量化文件
4. 匹配计算: 代码逻辑匹配回审稿人，线性相加得分数
"""

from typing import List, Optional, Union
from pathlib import Path
import json
import time

from config import Config
from data_loader import DataLoader, Paper
from feature_extractor import FeatureExtractor
from llm_quantifier import LLMQuantifier
from pros_cons_processor import ProsConsProcessor
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
        self.processor = ProsConsProcessor()
        self.analyzer = BiasAnalyzer()
        self.visualizer = Visualizer(output_dir)
        
        # 数据存储
        self.papers: List[Paper] = []
        self.analysis_results: List[BiasAnalysisResult] = []
        
        # 中间文件路径
        self.extraction_file: Optional[Path] = None
        self.anonymized_file: Optional[Path] = None
        self.mapping_file: Optional[Path] = None
        self.quantified_file: Optional[Path] = None
        
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
        logger.info("步骤 0: 数据加载")
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
    
    def load_paper_pdfs(self) -> 'ReviewBiasAnalysisPipeline':
        """
        为当前列表中的所有论文加载 PDF 内容（如果尚未加载）
        实现适配性优化：仅处理需要分析的论文
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"PDF提取: 正在为 {len(self.papers)} 篇待分析论文提取 PDF 文本")
        logger.info(f"{'='*70}")
        
        loaded_count = 0
        for paper in self.papers:
            if not paper.paper_content and paper.source_dir:
                self.data_loader._load_pdf_content(paper, paper.source_dir)
                if paper.paper_content:
                    loaded_count += 1
        
        logger.info(f"成功提取 {loaded_count} 篇论文的 PDF 内容")
        
        return self
    
    # ========== 步骤1: 特征提取 ==========
    
    def step1_extract_features(self) -> 'ReviewBiasAnalysisPipeline':
        """
        步骤1: 使用LLM提取每个审稿人的优缺点
        
        Returns:
            self（支持链式调用）
        """
        logger.info(f"\n{'='*70}")
        logger.info("步骤 1: 特征提取（独立提取每个审稿人的优缺点）")
        logger.info(f"{'='*70}")
        
        if not self.papers:
            raise ValueError("请先加载数据")
        
        # 提取优缺点
        self.papers = self.feature_extractor.extract_from_papers(self.papers)
        
        # 保存提取结果到文件
        self.extraction_file = self.feature_extractor.save_extraction_results(self.papers)
        
        # 显示摘要
        self.feature_extractor.display_extraction_summary(self.papers)
        
        return self
    
    # ========== 步骤2: 匿名化处理 ==========
    
    def step2_anonymize_and_shuffle(self) -> 'ReviewBiasAnalysisPipeline':
        """
        步骤2: 处理提取结果，去除审稿人信息，打乱顺序
        
        Returns:
            self（支持链式调用）
        """
        logger.info(f"\n{'='*70}")
        logger.info("步骤 2: 匿名化处理（去除审稿人信息，打乱顺序）")
        logger.info(f"{'='*70}")
        
        if not self.extraction_file:
            raise ValueError("请先执行步骤1（特征提取）")
        
        # 处理并输出匿名化文件
        self.anonymized_file, mapping = self.processor.process_extraction_file(
            self.extraction_file
        )
        
        # 保存映射文件路径
        self.mapping_file = Config.ANONYMIZED_DIR / "original_mapping.json"
        
        logger.info(f"✓ 匿名化完成，优缺点顺序已随机打乱")
        
        return self
    
    # ========== 步骤3: 权重量化 ==========
    
    def step3_quantify_weights(self) -> 'ReviewBiasAnalysisPipeline':
        """
        步骤3: 基于匿名化文件和PDF内容，使用LLM量化权重
        
        Returns:
            self（支持链式调用）
        """
        logger.info(f"\n{'='*70}")
        logger.info("步骤 3: 权重量化（基于匿名优缺点+论文全文）")
        logger.info(f"{'='*70}")
        
        if not self.anonymized_file:
            raise ValueError("请先执行步骤2（匿名化处理）")
        
        # 量化并输出结果
        self.quantified_file = self.quantifier.quantify_anonymized_file(
            self.anonymized_file,
            self.papers
        )
        
        return self
    
    # ========== 步骤4: 匹配并计算分数 ==========
    
    def step4_match_and_calculate(self) -> 'ReviewBiasAnalysisPipeline':
        """
        步骤4: 代码逻辑匹配回审稿人，线性相加得分数
        
        注意: 此步骤不使用LLM，完全使用代码逻辑
        
        Returns:
            self（支持链式调用）
        """
        logger.info(f"\n{'='*70}")
        logger.info("步骤 4: 匹配计算（代码逻辑匹配审稿人，线性相加）")
        logger.info(f"{'='*70}")
        
        if not self.quantified_file or not self.mapping_file:
            raise ValueError("请先执行步骤3（权重量化）")
        
        # 匹配并计算分数
        self.papers = self.processor.match_and_calculate_scores(
            self.quantified_file,
            self.mapping_file,
            self.papers
        )
        
        # 显示量化摘要
        self.quantifier.display_quantification_summary(self.papers)
        
        return self
    
    # ========== 批次处理 ==========
    
    def _split_into_batches(self, papers: List[Paper], batch_size: int) -> List[List[Paper]]:
        """将论文列表分割成批次"""
        batches = []
        for i in range(0, len(papers), batch_size):
            batches.append(papers[i:i + batch_size])
        return batches
    
    def _process_batch(
        self, 
        batch_papers: List[Paper], 
        batch_index: int, 
        total_batches: int
    ) -> tuple:
        """
        处理单个批次的论文（步骤1-3）
        
        Args:
            batch_papers: 当前批次的论文列表
            batch_index: 批次索引（从0开始）
            total_batches: 总批次数
            
        Returns:
            (extraction_data, anonymized_data, quantified_data, mapping_data)
        """
        batch_num = batch_index + 1
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 处理批次 {batch_num}/{total_batches} (共 {len(batch_papers)} 篇论文)")
        logger.info(f"{'='*70}")
        
        # 临时存储当前批次的论文
        original_papers = self.papers
        self.papers = batch_papers
        
        # 0. 提取PDF内容
        self.load_paper_pdfs()
        
        # 1. 特征提取
        logger.info(f"\n[批次{batch_num}] 步骤1: 特征提取")
        self.papers = self.feature_extractor.extract_from_papers(self.papers)
        
        # 收集提取数据
        extraction_data = []
        for paper in self.papers:
            paper_data = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "reviews": []
            }
            for review in paper.reviews:
                review_data = {
                    "reviewer_id": review.reviewer_id,
                    "actual_score": review.actual_score,
                    "pros": review.pros,
                    "cons": review.cons
                }
                paper_data["reviews"].append(review_data)
            extraction_data.append(paper_data)
        
        # 2. 匿名化处理（直接在内存中处理）
        logger.info(f"\n[批次{batch_num}] 步骤2: 匿名化处理")
        anonymized_data, mapping_data = self.processor.anonymize_in_memory(extraction_data)
        
        # 步骤之间添加延迟，避免API请求过于密集
        logger.info(f"  ⏳ 步骤间隔等待 {Config.BATCH_DELAY:.1f} 秒...")
        time.sleep(Config.BATCH_DELAY)
        
        # 3. 权重量化
        logger.info(f"\n[批次{batch_num}] 步骤3: 权重量化")
        quantified_data = self._quantify_batch_in_memory(anonymized_data, self.papers)
        
        # 恢复原始论文列表
        self.papers = original_papers
        
        logger.info(f"\n✓ 批次 {batch_num}/{total_batches} 处理完成")
        
        return extraction_data, anonymized_data, quantified_data, mapping_data
    
    def _quantify_batch_in_memory(
        self, 
        anonymized_data: List[dict], 
        papers: List[Paper]
    ) -> List[dict]:
        """在内存中量化单个批次，支持智能延迟"""
        from utils import safe_json_parse, ProgressTracker
        
        paper_content_map = {p.paper_id: p.paper_content for p in papers}
        quantified_results = []
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        for idx, paper_data in enumerate(anonymized_data):
            paper_id = paper_data["paper_id"]
            title = paper_data["title"]
            abstract = paper_data["abstract"]
            pros = paper_data["pros"]
            cons = paper_data["cons"]
            
            paper_content = paper_content_map.get(paper_id, "")
            
            logger.info(f"  [{idx+1}/{len(anonymized_data)}] 正在量化: {title[:45]}...")
            
            pros_count = len(pros)
            cons_count = len(cons)
            
            pros_text = "\n".join([
                f"{i+1}. [{p.get('category', '未分类')}] {p.get('description', '')}"
                for i, p in enumerate(pros)
            ]) if pros else "(无)"
            
            cons_text = "\n".join([
                f"{i+1}. [{c.get('category', '未分类')}] {c.get('description', '')}"
                for i, c in enumerate(cons)
            ]) if cons else "(无)"
            
            from config import PromptTemplates
            prompt = PromptTemplates.QUANTIFY_WEIGHTS.format(
                title=title,
                abstract=abstract,
                paper_content=paper_content[:15000] if paper_content else "(未提供论文全文)",
                pros_text=pros_text,
                cons_text=cons_text,
                pros_count=pros_count,
                cons_count=cons_count,
                min_score=Config.MIN_SCORE,
                max_score=Config.MAX_SCORE,
                base_score=Config.BASE_SCORE
            )
            
            try:
                response = self.quantifier._call_llm(prompt)
                result = safe_json_parse(response, default={
                    "pros_weights": [],
                    "cons_weights": [],
                    "expected_score_breakdown": {}
                })
                
                pros_weights = result.get("pros_weights", [])
                cons_weights = result.get("cons_weights", [])
                
                # 补齐缺失的权重
                while len(pros_weights) < pros_count:
                    i = len(pros_weights)
                    pros_weights.append({
                        "description": pros[i].get("description", "") if i < len(pros) else "",
                        "category": pros[i].get("category", "") if i < len(pros) else "",
                        "weight": 0.5,
                        "reasoning": "LLM未返回，使用默认值"
                    })
                
                while len(cons_weights) < cons_count:
                    i = len(cons_weights)
                    cons_weights.append({
                        "description": cons[i].get("description", "") if i < len(cons) else "",
                        "category": cons[i].get("category", "") if i < len(cons) else "",
                        "weight": -0.5,
                        "reasoning": "LLM未返回，使用默认值"
                    })
                
                quantified_results.append({
                    "paper_id": paper_id,
                    "title": title,
                    "pros_weights": pros_weights,
                    "cons_weights": cons_weights,
                    "expected_score_breakdown": result.get("expected_score_breakdown", {})
                })
                
                logger.info(f"    ✓ 完成: {len(pros_weights)} 优点, {len(cons_weights)} 缺点")
                consecutive_failures = 0
                
                # 智能延迟
                if idx < len(anonymized_data) - 1:
                    if (idx + 1) % 10 == 0:
                        # 每10篇长休息
                        logger.info(f"  ⏳ 长休息 {Config.BATCH_DELAY:.0f} 秒...")
                        time.sleep(Config.BATCH_DELAY)
                    else:
                        time.sleep(Config.REQUEST_DELAY)
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"  ✗ 量化失败: {e}")
                
                quantified_results.append({
                    "paper_id": paper_id,
                    "title": title,
                    "pros_weights": [
                        {"description": p.get("description", ""), "category": p.get("category", ""), "weight": 0, "reasoning": "量化失败"}
                        for p in pros
                    ],
                    "cons_weights": [
                        {"description": c.get("description", ""), "category": c.get("category", ""), "weight": 0, "reasoning": "量化失败"}
                        for c in cons
                    ],
                    "expected_score_breakdown": {}
                })
                
                # 连续失败时增加等待
                if consecutive_failures >= max_consecutive_failures:
                    wait_time = Config.BATCH_DELAY * 3
                    logger.warning(f"⚠️ 连续失败 {consecutive_failures} 次，等待 {wait_time:.0f} 秒...")
                    time.sleep(wait_time)
                    consecutive_failures = 0
        
        return quantified_results
    
    def _merge_batch_results(
        self,
        all_extraction: List[List[dict]],
        all_anonymized: List[List[dict]],
        all_quantified: List[List[dict]],
        all_mapping: List[dict]
    ) -> tuple:
        """合并所有批次的结果"""
        merged_extraction = []
        merged_anonymized = []
        merged_quantified = []
        merged_mapping = {}
        
        for batch_ext in all_extraction:
            merged_extraction.extend(batch_ext)
        
        for batch_anon in all_anonymized:
            merged_anonymized.extend(batch_anon)
        
        for batch_quant in all_quantified:
            merged_quantified.extend(batch_quant)
        
        for batch_map in all_mapping:
            merged_mapping.update(batch_map)
        
        return merged_extraction, merged_anonymized, merged_quantified, merged_mapping
    
    # ========== 完整流程 ==========
    
    def run_full_analysis(self, batch_size: int = None) -> dict:
        """
        运行完整的四步骤分析流程（支持批次处理）
        
        流程:
        1. 分批处理（每批10篇）:
           - 特征提取 -> 输出文件
           - 匿名化处理 -> 输出新文件
           - 权重量化 -> 输出量化文件
        2. 合并所有批次结果
        3. 匹配计算 -> 更新论文数据
        4. 偏差分析
        5. 可视化
        
        Args:
            batch_size: 每批处理的论文数量，默认为 Config.BATCH_SIZE
        
        Returns:
            分析结果摘要字典
        """
        batch_size = batch_size or Config.BATCH_SIZE
        
        logger.info("\n" + "="*70)
        logger.info(f"开始完整分析流程（批次处理，每批 {batch_size} 篇）")
        logger.info("="*70)
        
        # 分割成批次
        batches = self._split_into_batches(self.papers, batch_size)
        total_batches = len(batches)
        
        logger.info(f"共 {len(self.papers)} 篇论文，分为 {total_batches} 个批次处理")
        
        # 存储所有批次的结果
        all_extraction = []
        all_anonymized = []
        all_quantified = []
        all_mapping = []
        
        # 处理每个批次
        for i, batch_papers in enumerate(batches):
            extraction_data, anonymized_data, quantified_data, mapping_data = \
                self._process_batch(batch_papers, i, total_batches)
            
            all_extraction.append(extraction_data)
            all_anonymized.append(anonymized_data)
            all_quantified.append(quantified_data)
            all_mapping.append(mapping_data)
            
            # 批次之间的延迟
            if i < total_batches - 1:
                logger.info(f"\n⏳ 等待 {Config.BATCH_DELAY} 秒后处理下一批次...")
                time.sleep(Config.BATCH_DELAY)
        
        # 合并所有批次的结果
        logger.info(f"\n{'='*70}")
        logger.info("合并所有批次结果")
        logger.info(f"{'='*70}")
        
        merged_extraction, merged_anonymized, merged_quantified, merged_mapping = \
            self._merge_batch_results(all_extraction, all_anonymized, all_quantified, all_mapping)
        
        # 保存合并后的结果到文件
        self.extraction_file = Config.EXTRACTION_DIR / "extraction_results.json"
        with open(self.extraction_file, 'w', encoding='utf-8') as f:
            json.dump(merged_extraction, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存提取结果: {self.extraction_file}")
        
        self.anonymized_file = Config.ANONYMIZED_DIR / "anonymized_pros_cons.json"
        with open(self.anonymized_file, 'w', encoding='utf-8') as f:
            json.dump(merged_anonymized, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存匿名化结果: {self.anonymized_file}")
        
        self.mapping_file = Config.ANONYMIZED_DIR / "original_mapping.json"
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(merged_mapping, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存映射文件: {self.mapping_file}")
        
        self.quantified_file = Config.QUANTIFIED_DIR / "quantified_weights.json"
        with open(self.quantified_file, 'w', encoding='utf-8') as f:
            json.dump(merged_quantified, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存量化结果: {self.quantified_file}")
        
        # 4. 匹配计算（使用合并后的文件）
        self.step4_match_and_calculate()
        
        # 5. 显示摘要
        self.feature_extractor.display_extraction_summary(self.papers)
        self.quantifier.display_quantification_summary(self.papers)
        
        # 6. 保存每篇论文的详细报告
        self.save_individual_reports()
        
        # 7. 偏差分析
        self.analyze_bias()
        
        # 8. 生成可视化
        self.generate_visualizations()
        
        # 9. 生成摘要
        summary = {
            "data_statistics": self.data_loader.get_statistics(),
            "extraction_summary": self.feature_extractor.get_extraction_summary(self.papers),
            "quantification_summary": self.quantifier.get_quantification_summary(self.papers),
            "bias_statistics": self.analyzer.global_statistics(self.analysis_results),
            "batch_info": {
                "batch_size": batch_size,
                "total_batches": total_batches,
                "total_papers": len(self.papers)
            },
            "output_files": {
                "extraction": str(self.extraction_file),
                "anonymized": str(self.anonymized_file),
                "mapping": str(self.mapping_file),
                "quantified": str(self.quantified_file)
            }
        }
        
        logger.info("\n" + "="*70)
        logger.info("完整分析流程已完成！")
        logger.info("="*70)
        logger.info(f"\n批次处理信息:")
        logger.info(f"  每批大小: {batch_size} 篇")
        logger.info(f"  总批次数: {total_batches}")
        logger.info(f"  总论文数: {len(self.papers)}")
        logger.info(f"\n中间文件:")
        logger.info(f"  步骤1 提取结果: {self.extraction_file}")
        logger.info(f"  步骤2 匿名化文件: {self.anonymized_file}")
        logger.info(f"  步骤2 映射文件: {self.mapping_file}")
        logger.info(f"  步骤3 量化结果: {self.quantified_file}")
        
        return summary
    
    # ========== 偏差分析 ==========
    
    def analyze_bias(self) -> List[BiasAnalysisResult]:
        """
        分析偏差
        
        Returns:
            偏差分析结果列表
        """
        logger.info(f"\n{'='*70}")
        logger.info("步骤 5: 偏差分析")
        logger.info(f"{'='*70}")
        
        if not self.papers:
            raise ValueError("请先加载数据")
        
        # 检查是否已计算期望分数
        if self.papers[0].reviews[0].expected_score is None:
            logger.warning("未检测到期望分数，请先执行步骤1-4")
            return []
        
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
        logger.info("步骤 6: 生成可视化")
        logger.info(f"{'='*70}")
        
        if not self.analysis_results:
            logger.warning("未检测到分析结果，跳过可视化")
            return self
        
        self.visualizer.generate_all_plots(self.papers, self.analysis_results)
        
        return self
    
    # ========== 结果保存 ==========
    
    def save_individual_reports(self, output_dir: Optional[Union[str, Path]] = None):
        """
        为每篇论文保存详细的分析报告
        
        Args:
            output_dir: 输出目录，默认为 Config.DETAILS_DIR
        """
        logger.info(f"\n{'='*70}")
        logger.info("保存每篇论文的详细分析报告")
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
        ]
        
        # 添加论文全文信息
        if paper.paper_content:
            content.append(f"\n## 论文全文内容 (截取前500字)\n{paper.paper_content[:500]}...")
        else:
            content.append("\n## 论文全文内容\n(未提取到论文全文)")
        
        content.append(f"\n## 各审稿人详细分析\n")
        
        for i, review in enumerate(paper.reviews):
            content.append(f"### 审稿人 {review.reviewer_id}")
            content.append(f"- **实际分数:** {review.actual_score}")
            content.append(f"- **期望分数:** {review.expected_score:.2f}" if review.expected_score is not None else "- **期望分数:** 未计算")
            content.append(f"- **偏差 (实际 - 期望):** {review.bias:+.2f}" if review.bias is not None else "- **偏差:** 未计算")
            
            # 优点
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
            
            # 缺点
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
    for i in range(2):
        paper = Paper(
            paper_id=f"paper_{i}",
            title=f"Test Paper {i}: An Innovative Approach",
            abstract="This paper presents a novel method for solving complex problems.",
            paper_content="Full paper content here..."
        )
        
        for j in range(2):
            review = Review(
                reviewer_id=f"reviewer_{j}",
                review_text=f"""
                This paper has several strengths and weaknesses.
                
                Strengths:
                - Novel approach to the problem
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
# pipeline.step1_extract_features()
# pipeline.step2_anonymize_and_shuffle()
# pipeline.step3_quantify_weights()
# pipeline.step4_match_and_calculate()
# pipeline.analyze_bias()
# pipeline.generate_visualizations()

# 保存结果
# pipeline.save_results()
# pipeline.generate_report()
    """)
    
    # 清理
    test_data_file.unlink()
