"""
特征提取模块
使用LLM从审稿意见中提取结构化的优缺点
"""

from typing import List, Dict, Any
from openai import OpenAI

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
    
    def extract_pros_cons_from_review(
        self, 
        review: Review, 
        paper_title: str, 
        paper_abstract: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        使用LLM从单条审稿意见中提取优缺点
        
        Args:
            review: 审稿记录
            paper_title: 论文标题
            paper_abstract: 论文摘要
            
        Returns:
            包含pros和cons的字典
        """
        logger.debug(f"正在提取审稿人 {review.reviewer_id} 的优缺点...")
        
        # 构建提示词
        categories_str = ", ".join(Config.CATEGORIES)
        prompt = PromptTemplates.EXTRACT_PROS_CONS.format(
            title=paper_title,
            abstract=paper_abstract,
            review_text=review.review_text,
            categories=categories_str
        )
        
        # 调用LLM
        try:
            response = self._call_llm(prompt)
            
            # 解析JSON响应
            result = safe_json_parse(response, default={
                "pros": [],
                "cons": []
            })
            
            # 验证结果
            if "pros" not in result:
                result["pros"] = []
            if "cons" not in result:
                result["cons"] = []
            
            # 确保每个优缺点都有必需的字段
            for pro in result["pros"]:
                if "category" not in pro:
                    pro["category"] = "未分类"
                if "description" not in pro:
                    pro["description"] = ""
            
            for con in result["cons"]:
                if "category" not in con:
                    con["category"] = "未分类"
                if "description" not in con:
                    con["description"] = ""
            
            logger.debug(
                f"提取到 {len(result['pros'])} 个优点，"
                f"{len(result['cons'])} 个缺点"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LLM提取优缺点失败: {e}")
            # 返回空结果
            return {"pros": [], "cons": []}
    
    def extract_from_paper(self, paper: Paper) -> Paper:
        """
        从论文的所有审稿意见中提取优缺点
        
        Args:
            paper: 论文对象（会被原地修改）
            
        Returns:
            更新后的论文对象
        """
        logger.info(f"正在处理论文: {paper.title}")
        
        for review in paper.reviews:
            try:
                result = self.extract_pros_cons_from_review(
                    review=review,
                    paper_title=paper.title,
                    paper_abstract=paper.abstract
                )
                
                # 更新review对象
                review.pros = result["pros"]
                review.cons = result["cons"]
                
                logger.info(
                    f"  审稿人 {review.reviewer_id}: "
                    f"{len(review.pros)} 优点, {len(review.cons)} 缺点"
                )
                
            except Exception as e:
                logger.error(
                    f"处理审稿人 {review.reviewer_id} 时出错: {e}"
                )
                # 设置空结果
                review.pros = []
                review.cons = []
        
        return paper
    
    def extract_from_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        批量提取多篇论文的优缺点
        
        Args:
            papers: 论文列表（会被原地修改）
            
        Returns:
            更新后的论文列表
        """
        logger.info(f"开始批量提取 {len(papers)} 篇论文的优缺点...")
        
        tracker = ProgressTracker(
            total=sum(len(p.reviews) for p in papers),
            description="特征提取"
        )
        
        for paper in papers:
            self.extract_from_paper(paper)
            tracker.update(len(paper.reviews))
        
        tracker.finish()
        
        return papers
    
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
        
        print("\n✓ 特征提取器测试完成！")
        
    except ValueError as e:
        print(f"\n⚠ 需要配置API密钥才能测试: {e}")


