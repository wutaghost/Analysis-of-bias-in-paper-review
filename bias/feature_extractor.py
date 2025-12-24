"""
特征提取模块
使用LLM从审稿意见中提取结构化的优缺点
步骤1: 独立提取每个审稿人的优缺点并保存到文件
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from openai import OpenAI

import time
from config import Config, PromptTemplates
from data_loader import Paper, Review
from utils import logger, cached, retry_on_failure, safe_json_parse, ProgressTracker


class FeatureExtractor:
    """特征提取器 - 从审稿文本中提取优缺点"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        初始化特征提取器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL（可选）
            model: 使用的模型名称
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.base_url = base_url or Config.OPENAI_BASE_URL
        self.model = model or Config.MODEL_NAME
        
        # 初始化OpenAI客户端
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.client = OpenAI(**client_kwargs)
        
        logger.info(f"特征提取器已初始化，使用模型: {self.model}")
    
    @retry_on_failure()
    @cached
    def _call_llm(self, prompt: str) -> str:
        """
        调用LLM API
        
        Args:
            prompt: 输入提示词
            
        Returns:
            LLM响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位资深的学术论文审稿专家，擅长分析审稿意见。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise
    
    def extract_pros_cons_from_paper(
        self, 
        paper: Paper
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        使用LLM一次性提取一篇论文所有审稿意见的优缺点
        
        Args:
            paper: 论文对象（包含所有审稿意见）
            
        Returns:
            包含每个审稿人优缺点的字典
        """
        logger.info(f"正在提取论文 {paper.title[:50]}... 的所有审稿优缺点")
        
        # 构建所有审稿意见的文本
        all_reviews_text = ""
        for i, review in enumerate(paper.reviews):
            all_reviews_text += f"\n{'='*40}\n"
            all_reviews_text += f"【审稿人 {review.reviewer_id}】\n"
            all_reviews_text += f"{'='*40}\n"
            all_reviews_text += f"{review.review_text}\n"
        
        # 构建提示词
        categories_str = ", ".join(Config.CATEGORIES)
        prompt = PromptTemplates.EXTRACT_PROS_CONS_BATCH.format(
            title=paper.title,
            abstract=paper.abstract,
            num_reviewers=len(paper.reviews),
            all_reviews_text=all_reviews_text,
            categories=categories_str
        )
        
        # 调用LLM（一次调用处理所有审稿意见）
        try:
            response = self._call_llm(prompt)
            
            # 解析JSON响应
            result = safe_json_parse(response, default={"reviewers": []})
            
            # 构建 reviewer_id -> 优缺点 的映射
            reviewer_results = {}
            for reviewer_data in result.get("reviewers", []):
                reviewer_id = reviewer_data.get("reviewer_id", "")
                pros = reviewer_data.get("pros", [])
                cons = reviewer_data.get("cons", [])
                
                # 确保每个优缺点都有必需的字段
                for pro in pros:
                    if "category" not in pro:
                        pro["category"] = "未分类"
                    if "description" not in pro:
                        pro["description"] = ""
                
                for con in cons:
                    if "category" not in con:
                        con["category"] = "未分类"
                    if "description" not in con:
                        con["description"] = ""
                
                reviewer_results[reviewer_id] = {
                    "pros": pros,
                    "cons": cons
                }
            
            # 为每个审稿人更新结果
            for review in paper.reviews:
                if review.reviewer_id in reviewer_results:
                    data = reviewer_results[review.reviewer_id]
                    review.pros = data["pros"]
                    review.cons = data["cons"]
                else:
                    # 如果LLM没有返回该审稿人的结果，设为空
                    logger.warning(f"  ⚠ 未找到审稿人 {review.reviewer_id} 的提取结果")
                    review.pros = []
                    review.cons = []
                
                logger.info(
                    f"  审稿人 {review.reviewer_id}: "
                    f"{len(review.pros)} 优点, {len(review.cons)} 缺点"
                )
            
            return reviewer_results
            
        except Exception as e:
            logger.error(f"LLM提取优缺点失败: {e}")
            # 为所有审稿人设置空结果
            for review in paper.reviews:
                review.pros = []
                review.cons = []
            return {}
    
    def extract_from_paper(self, paper: Paper) -> Paper:
        """
        从论文的所有审稿意见中提取优缺点（一次LLM调用）
        
        Args:
            paper: 论文对象（会被原地修改）
            
        Returns:
            更新后的论文对象
        """
        # 一次性提取该论文所有审稿人的优缺点
        self.extract_pros_cons_from_paper(paper)
        return paper
    
    def extract_from_papers(self, papers: List[Paper], checkpoint_interval: int = 5) -> List[Paper]:
        """
        批量提取多篇论文的优缺点
        每篇论文只需要一次LLM调用（而不是每个审稿人一次）
        支持断点续传：跳过已有提取结果的论文
        
        Args:
            papers: 论文列表（会被原地修改）
            checkpoint_interval: 每隔多少篇保存一次检查点
            
        Returns:
            更新后的论文列表
        """
        total_reviews = sum(len(p.reviews) for p in papers)
        logger.info(f"开始批量特征提取，共 {len(papers)} 篇论文 ({total_reviews} 条审稿)")
        logger.info(f"📌 优化: 每篇论文1次API调用，共需 {len(papers)} 次API调用")
        
        # 检查已有提取结果的论文（断点续传）
        papers_to_process = []
        papers_already_done = 0
        for paper in papers:
            # 检查是否所有审稿都已有提取结果
            has_results = all(
                len(r.pros) > 0 or len(r.cons) > 0 
                for r in paper.reviews
            )
            if has_results:
                papers_already_done += 1
            else:
                papers_to_process.append(paper)
        
        if papers_already_done > 0:
            logger.info(f"📋 断点续传: 跳过 {papers_already_done} 篇已处理论文")
        
        if not papers_to_process:
            logger.info("所有论文已处理完成，无需API调用")
            return papers
        
        logger.info(f"需要处理: {len(papers_to_process)} 篇论文")
        
        tracker = ProgressTracker(
            total=len(papers_to_process),
            description="特征提取"
        )
        
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        for i, paper in enumerate(papers_to_process):
            try:
                self.extract_from_paper(paper)
                tracker.update(1)
                consecutive_failures = 0  # 重置连续失败计数
                
                # 每隔一定数量保存检查点
                if (i + 1) % checkpoint_interval == 0:
                    logger.info(f"💾 检查点: 已处理 {i + 1}/{len(papers_to_process)} 篇")
                
                # 论文之间添加延迟，避免API请求过于密集
                if i < len(papers_to_process) - 1:
                    # 根据处理进度动态调整延迟
                    if i > 0 and i % 10 == 0:
                        # 每处理10篇，增加一次长休息
                        long_delay = Config.BATCH_DELAY
                        logger.info(f"  ⏳ 长休息 {long_delay:.1f} 秒 (已处理 {i+1} 篇)...")
                        time.sleep(long_delay)
                    else:
                        logger.info(f"  ⏳ 等待 {Config.REQUEST_DELAY:.1f} 秒...")
                        time.sleep(Config.REQUEST_DELAY)
                        
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"处理论文 {paper.title[:30]}... 失败: {e}")
                
                # 如果连续失败多次，增加等待时间
                if consecutive_failures >= max_consecutive_failures:
                    wait_time = Config.BATCH_DELAY * 2
                    logger.warning(f"⚠️ 连续失败 {consecutive_failures} 次，等待 {wait_time} 秒后继续...")
                    time.sleep(wait_time)
                    consecutive_failures = 0
        
        tracker.finish()
        
        return papers
    
    def save_extraction_results(self, papers: List[Paper], output_dir: Path = None) -> Path:
        """
        步骤1: 将提取结果保存到文件
        保存格式: 每个审稿人的优缺点，包含审稿人ID信息
        
        Args:
            papers: 论文列表（已提取优缺点）
            output_dir: 输出目录，默认为 Config.EXTRACTION_DIR
            
        Returns:
            输出文件路径
        """
        output_dir = output_dir or Config.EXTRACTION_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extraction_data = []
        
        for paper in papers:
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
        
        output_file = output_dir / "extraction_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"特征提取结果已保存到: {output_file}")
        return output_file
    
    def compare_reviews_similarity(
        self,
        review1: Review,
        review2: Review,
        paper_title: str
    ) -> Dict[str, Any]:
        """
        比较两个审稿意见的相似度
        
        Args:
            review1: 第一个审稿记录
            review2: 第二个审稿记录
            paper_title: 论文标题
            
        Returns:
            相似度分析结果
        """
        # 格式化优缺点
        pros_1 = "\n".join([f"- {p['description']}" for p in review1.pros])
        cons_1 = "\n".join([f"- {c['description']}" for c in review1.cons])
        pros_2 = "\n".join([f"- {p['description']}" for p in review2.pros])
        cons_2 = "\n".join([f"- {c['description']}" for c in review2.cons])
        
        # 构建提示词
        prompt = PromptTemplates.COMPARE_REVIEWS.format(
            title=paper_title,
            pros_1=pros_1 or "(无)",
            cons_1=cons_1 or "(无)",
            pros_2=pros_2 or "(无)",
            cons_2=cons_2 or "(无)"
        )
        
        # 调用LLM
        response = self._call_llm(prompt)
        
        # 解析响应
        result = safe_json_parse(response, default={
            "pros_similarity": 0.0,
            "cons_similarity": 0.0,
            "overall_similarity": 0.0,
            "common_pros": [],
            "common_cons": [],
            "unique_to_reviewer1": [],
            "unique_to_reviewer2": []
        })
        
        return result
    
    def analyze_paper_review_similarity(self, paper: Paper) -> List[Dict[str, Any]]:
        """
        分析一篇论文的所有审稿意见之间的相似度
        
        Args:
            paper: 论文对象
            
        Returns:
            相似度分析结果列表
        """
        if len(paper.reviews) < 2:
            logger.warning(f"论文 {paper.title} 的审稿数少于2，跳过相似度分析")
            return []
        
        logger.info(f"分析论文 {paper.title} 的审稿相似度...")
        
        similarities = []
        reviews = paper.reviews
        
        # 两两比较
        for i in range(len(reviews)):
            for j in range(i + 1, len(reviews)):
                try:
                    similarity = self.compare_reviews_similarity(
                        review1=reviews[i],
                        review2=reviews[j],
                        paper_title=paper.title
                    )
                    
                    similarity["reviewer1_id"] = reviews[i].reviewer_id
                    similarity["reviewer2_id"] = reviews[j].reviewer_id
                    similarity["score1"] = reviews[i].actual_score
                    similarity["score2"] = reviews[j].actual_score
                    similarity["score_diff"] = abs(
                        reviews[i].actual_score - reviews[j].actual_score
                    )
                    
                    similarities.append(similarity)
                    
                    logger.info(
                        f"  {reviews[i].reviewer_id} vs {reviews[j].reviewer_id}: "
                        f"相似度={similarity['overall_similarity']:.2f}, "
                        f"分数差={similarity['score_diff']:.1f}"
                    )
                    
                except Exception as e:
                    logger.error(f"比较审稿意见时出错: {e}")
        
        return similarities
    
    def get_extraction_summary(self, papers: List[Paper]) -> Dict[str, Any]:
        """
        获取特征提取的摘要统计
        
        Args:
            papers: 论文列表
            
        Returns:
            统计摘要
        """
        total_reviews = sum(len(p.reviews) for p in papers)
        total_pros = sum(
            len(r.pros) for p in papers for r in p.reviews
        )
        total_cons = sum(
            len(r.cons) for p in papers for r in p.reviews
        )
        
        # 统计各类别的数量
        category_stats = {cat: {"pros": 0, "cons": 0} for cat in Config.CATEGORIES}
        
        for paper in papers:
            for review in paper.reviews:
                for pro in review.pros:
                    cat = pro.get("category", "其他")
                    if cat in category_stats:
                        category_stats[cat]["pros"] += 1
                
                for con in review.cons:
                    cat = con.get("category", "其他")
                    if cat in category_stats:
                        category_stats[cat]["cons"] += 1
        
        summary = {
            "total_papers": len(papers),
            "total_reviews": total_reviews,
            "total_pros": total_pros,
            "total_cons": total_cons,
            "avg_pros_per_review": total_pros / total_reviews if total_reviews > 0 else 0,
            "avg_cons_per_review": total_cons / total_reviews if total_reviews > 0 else 0,
            "category_distribution": category_stats
        }
        
        return summary
    
    def display_extraction_summary(self, papers: List[Paper]):
        """显示特征提取摘要"""
        summary = self.get_extraction_summary(papers)
        
        print("\n" + "=" * 50)
        print("特征提取摘要")
        print("=" * 50)
        print(f"处理论文数: {summary['total_papers']}")
        print(f"审稿记录数: {summary['total_reviews']}")
        print(f"提取优点总数: {summary['total_pros']}")
        print(f"提取缺点总数: {summary['total_cons']}")
        print(f"平均每条审稿的优点数: {summary['avg_pros_per_review']:.2f}")
        print(f"平均每条审稿的缺点数: {summary['avg_cons_per_review']:.2f}")
        
        print("\n各类别分布:")
        for cat, counts in summary['category_distribution'].items():
            total = counts['pros'] + counts['cons']
            if total > 0:
                print(f"  {cat}:")
                print(f"    优点: {counts['pros']}, 缺点: {counts['cons']}")
        
        print("=" * 50 + "\n")


if __name__ == "__main__":
    # 测试特征提取器
    from data_loader import DataLoader, Paper, Review
    
    # 创建测试数据
    test_review = Review(
        reviewer_id="test_reviewer",
        review_text="""
        This paper presents an interesting approach to neural machine translation.
        
        Strengths:
        - The proposed attention mechanism is novel and effective
        - Experimental results are comprehensive
        - The paper is well-written
        
        Weaknesses:
        - Missing comparisons with recent SOTA methods
        - The computational cost analysis is insufficient
        - Some implementation details are unclear
        """,
        actual_score=7.0
    )
    
    test_paper = Paper(
        paper_id="test_paper",
        title="Attention Is All You Need",
        abstract="We propose a new neural network architecture based on attention mechanisms."
    )
    test_paper.add_review(test_review)
    
    # 测试提取
    extractor = FeatureExtractor()
    
    try:
        Config.validate()
        extractor.extract_from_paper(test_paper)
        
        print("\n提取结果:")
        print(f"优点数: {len(test_review.pros)}")
        print(f"缺点数: {len(test_review.cons)}")
        
        # 测试保存
        extractor.save_extraction_results([test_paper])
        
        print("\n✓ 特征提取器测试完成！")
        
    except ValueError as e:
        print(f"\n⚠ 需要配置API密钥才能测试: {e}")
