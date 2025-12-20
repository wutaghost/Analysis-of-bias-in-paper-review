"""
LLM分值量化模块
使用LLM为每个优缺点赋予量化权重
"""

from typing import List, Dict, Any
from openai import OpenAI

from config import Config, PromptTemplates
from data_loader import Paper, Review
from utils import logger, cached, retry_on_failure, safe_json_parse, ProgressTracker


class LLMQuantifier:
    """LLM量化器 - 为优缺点赋予权重分数"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        初始化量化器
        
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
        
        logger.info(f"LLM量化器已初始化，使用模型: {self.model}")
    
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
                        "content": "你是一位客观公正的学术评审专家，擅长量化分析论文的优缺点。"
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
    
    def quantify_review(
        self,
        review: Review,
        paper_title: str,
        paper_abstract: str
    ) -> Dict[str, Any]:
        """
        为单条审稿意见的优缺点赋予权重
        
        Args:
            review: 审稿记录（必须已提取pros和cons）
            paper_title: 论文标题
            paper_abstract: 论文摘要
            
        Returns:
            包含权重信息的字典
        """
        # 检查是否已提取优缺点
        if not hasattr(review, 'pros') or not hasattr(review, 'cons'):
            raise ValueError("Review必须先通过FeatureExtractor提取优缺点")
        
        if not review.pros and not review.cons:
            logger.warning(f"审稿人 {review.reviewer_id} 没有优缺点，跳过量化")
            return {
                "pros_weights": [],
                "cons_weights": [],
                "expected_score_breakdown": {
                    "base_score": Config.BASE_SCORE,
                    "total_pros_weight": 0.0,
                    "total_cons_weight": 0.0,
                    "expected_score": Config.BASE_SCORE
                }
            }
        
        # 格式化优缺点文本
        pros_text = "\n".join([
            f"{i+1}. [{p.get('category', '未分类')}] {p.get('description', '')}"
            for i, p in enumerate(review.pros)
        ]) if review.pros else "(无)"
        
        cons_text = "\n".join([
            f"{i+1}. [{c.get('category', '未分类')}] {c.get('description', '')}"
            for i, c in enumerate(review.cons)
        ]) if review.cons else "(无)"
        
        # 构建提示词
        prompt = PromptTemplates.QUANTIFY_WEIGHTS.format(
            title=paper_title,
            abstract=paper_abstract,
            pros_text=pros_text,
            cons_text=cons_text,
            min_score=Config.MIN_SCORE,
            max_score=Config.MAX_SCORE,
            base_score=Config.BASE_SCORE
        )
        
        # 调用LLM
        logger.debug(f"正在量化审稿人 {review.reviewer_id} 的优缺点权重...")
        response = self._call_llm(prompt)
        
        # 解析JSON响应
        result = safe_json_parse(response, default={
            "pros_weights": [],
            "cons_weights": [],
            "expected_score_breakdown": {
                "base_score": Config.BASE_SCORE,
                "total_pros_weight": 0.0,
                "total_cons_weight": 0.0,
                "expected_score": Config.BASE_SCORE
            }
        })
        
        # 验证和补充结果
        if "pros_weights" not in result:
            result["pros_weights"] = []
        if "cons_weights" not in result:
            result["cons_weights"] = []
        
        # 确保权重数量匹配
        if len(result["pros_weights"]) != len(review.pros):
            logger.warning(
                f"优点权重数量不匹配: 期望{len(review.pros)}, "
                f"实际{len(result['pros_weights'])}"
            )
        
        if len(result["cons_weights"]) != len(review.cons):
            logger.warning(
                f"缺点权重数量不匹配: 期望{len(review.cons)}, "
                f"实际{len(result['cons_weights'])}"
            )
        
        # 计算期望分数
        if "expected_score_breakdown" not in result:
            total_pros = sum(
                w.get("weight", 0) for w in result["pros_weights"]
            )
            total_cons = sum(
                w.get("weight", 0) for w in result["cons_weights"]
            )
            expected = Config.BASE_SCORE + total_pros + total_cons
            
            # 确保分数在合理范围内
            expected = max(Config.MIN_SCORE, min(Config.MAX_SCORE, expected))
            
            result["expected_score_breakdown"] = {
                "base_score": Config.BASE_SCORE,
                "total_pros_weight": total_pros,
                "total_cons_weight": total_cons,
                "expected_score": expected
            }
        
        logger.debug(
            f"量化完成: 期望分数={result['expected_score_breakdown']['expected_score']:.2f}, "
            f"实际分数={review.actual_score:.2f}"
        )
        
        return result
    
    def quantify_paper(self, paper: Paper) -> Paper:
        """
        为论文的所有审稿意见量化权重
        
        Args:
            paper: 论文对象（会被原地修改）
            
        Returns:
            更新后的论文对象
        """
        logger.info(f"正在量化论文: {paper.title}")
        
        for review in paper.reviews:
            try:
                result = self.quantify_review(
                    review=review,
                    paper_title=paper.title,
                    paper_abstract=paper.abstract
                )
                
                # 更新review对象
                review.pros_weights = result["pros_weights"]
                review.cons_weights = result["cons_weights"]
                review.expected_score = result["expected_score_breakdown"]["expected_score"]
                review.bias = review.actual_score - review.expected_score
                
                logger.info(
                    f"  审稿人 {review.reviewer_id}: "
                    f"期望={review.expected_score:.2f}, "
                    f"实际={review.actual_score:.2f}, "
                    f"偏差={review.bias:+.2f}"
                )
                
            except Exception as e:
                logger.error(
                    f"处理审稿人 {review.reviewer_id} 时出错: {e}"
                )
                # 设置默认值
                review.pros_weights = []
                review.cons_weights = []
                review.expected_score = Config.BASE_SCORE
                review.bias = review.actual_score - Config.BASE_SCORE
        
        return paper
    
    def quantify_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        批量量化多篇论文的权重
        
        Args:
            papers: 论文列表（会被原地修改）
            
        Returns:
            更新后的论文列表
        """
        logger.info(f"开始批量量化 {len(papers)} 篇论文的权重...")
        
        tracker = ProgressTracker(
            total=sum(len(p.reviews) for p in papers),
            description="权重量化"
        )
        
        for paper in papers:
            self.quantify_paper(paper)
            tracker.update(len(paper.reviews))
        
        tracker.finish()
        
        return papers
    
    def calculate_expected_score(
        self,
        pros_weights: List[Dict[str, Any]],
        cons_weights: List[Dict[str, Any]],
        base_score: float = None
    ) -> float:
        """
        根据权重计算期望分数
        
        Args:
            pros_weights: 优点权重列表
            cons_weights: 缺点权重列表
            base_score: 基准分数
            
        Returns:
            期望分数
        """
        base = base_score if base_score is not None else Config.BASE_SCORE
        
        total_pros = sum(w.get("weight", 0) for w in pros_weights)
        total_cons = sum(w.get("weight", 0) for w in cons_weights)
        
        expected = base + total_pros + total_cons
        
        # 限制在合理范围内
        expected = max(Config.MIN_SCORE, min(Config.MAX_SCORE, expected))
        
        return expected
    
    def get_quantification_summary(self, papers: List[Paper]) -> Dict[str, Any]:
        """
        获取量化结果的摘要统计
        
        Args:
            papers: 论文列表
            
        Returns:
            统计摘要
        """
        all_biases = []
        all_expected_scores = []
        all_actual_scores = []
        
        for paper in papers:
            for review in paper.reviews:
                if review.expected_score is not None:
                    all_expected_scores.append(review.expected_score)
                    all_actual_scores.append(review.actual_score)
                    if review.bias is not None:
                        all_biases.append(review.bias)
        
        import numpy as np
        
        summary = {
            "total_reviews": len(all_actual_scores),
            "bias_stats": {
                "mean": float(np.mean(all_biases)) if all_biases else 0,
                "std": float(np.std(all_biases)) if all_biases else 0,
                "median": float(np.median(all_biases)) if all_biases else 0,
                "min": float(np.min(all_biases)) if all_biases else 0,
                "max": float(np.max(all_biases)) if all_biases else 0,
            },
            "expected_score_stats": {
                "mean": float(np.mean(all_expected_scores)) if all_expected_scores else 0,
                "std": float(np.std(all_expected_scores)) if all_expected_scores else 0,
            },
            "actual_score_stats": {
                "mean": float(np.mean(all_actual_scores)) if all_actual_scores else 0,
                "std": float(np.std(all_actual_scores)) if all_actual_scores else 0,
            },
            "correlation": float(np.corrcoef(
                all_expected_scores, all_actual_scores
            )[0, 1]) if len(all_expected_scores) > 1 else 0
        }
        
        return summary
    
    def display_quantification_summary(self, papers: List[Paper]):
        """显示量化结果摘要"""
        summary = self.get_quantification_summary(papers)
        
        print("\n" + "=" * 50)
        print("权重量化摘要")
        print("=" * 50)
        print(f"处理审稿数: {summary['total_reviews']}")
        
        print("\n偏差统计 (Bias = Actual - Expected):")
        bias_stats = summary['bias_stats']
        print(f"  平均值: {bias_stats['mean']:+.2f}")
        print(f"  标准差: {bias_stats['std']:.2f}")
        print(f"  中位数: {bias_stats['median']:+.2f}")
        print(f"  范围: [{bias_stats['min']:+.2f}, {bias_stats['max']:+.2f}]")
        
        print("\n期望分数统计:")
        exp_stats = summary['expected_score_stats']
        print(f"  平均值: {exp_stats['mean']:.2f}")
        print(f"  标准差: {exp_stats['std']:.2f}")
        
        print("\n实际分数统计:")
        act_stats = summary['actual_score_stats']
        print(f"  平均值: {act_stats['mean']:.2f}")
        print(f"  标准差: {act_stats['std']:.2f}")
        
        print(f"\n期望分数与实际分数的相关系数: {summary['correlation']:.3f}")
        
        print("=" * 50 + "\n")


if __name__ == "__main__":
    # 测试量化器
    from data_loader import Paper, Review
    
    # 创建测试数据
    test_review = Review(
        reviewer_id="test_reviewer",
        review_text="Test review",
        actual_score=7.0
    )
    
    # 模拟已提取的优缺点
    test_review.pros = [
        {"description": "创新的方法", "category": "创新性 (Novelty/Originality)"},
        {"description": "实验充分", "category": "实验充分性 (Experimental Rigor)"}
    ]
    test_review.cons = [
        {"description": "写作有瑕疵", "category": "写作质量 (Writing Quality)"}
    ]
    
    test_paper = Paper(
        paper_id="test_paper",
        title="Test Paper",
        abstract="This is a test abstract."
    )
    test_paper.add_review(test_review)
    
    # 测试量化
    quantifier = LLMQuantifier()
    
    try:
        Config.validate()
        quantifier.quantify_paper(test_paper)
        
        print("\n量化结果:")
        print(f"期望分数: {test_review.expected_score:.2f}")
        print(f"实际分数: {test_review.actual_score:.2f}")
        print(f"偏差: {test_review.bias:+.2f}")
        
        print("\n✓ LLM量化器测试完成！")
        
    except ValueError as e:
        print(f"\n⚠ 需要配置API密钥才能测试: {e}")




