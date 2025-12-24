"""
LLM分值量化模块
步骤3: 使用LLM为匿名化后的优缺点赋予量化权重
"""

import json
import time
from typing import List, Dict, Any
from pathlib import Path
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
    
    def quantify_anonymized_file(
        self,
        anonymized_file: Path,
        papers: List[Paper],
        output_dir: Path = None
    ) -> Path:
        """
        步骤3: 对匿名化的优缺点进行量化
        
        Args:
            anonymized_file: 步骤2输出的匿名化文件
            papers: 论文列表（用于获取PDF内容）
            output_dir: 输出目录，默认为 Config.QUANTIFIED_DIR
            
        Returns:
            量化结果文件路径
        """
        output_dir = output_dir or Config.QUANTIFIED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"步骤3: 开始量化匿名优缺点")
        logger.info(f"读取匿名化文件: {anonymized_file}")
        
        # 读取匿名化数据
        with open(anonymized_file, 'r', encoding='utf-8') as f:
            anonymized_data = json.load(f)
        
        # 构建论文内容索引
        paper_content_map = {p.paper_id: p.paper_content for p in papers}
        
        quantified_results = []
        
        tracker = ProgressTracker(
            total=len(anonymized_data),
            description="权重量化"
        )
        
        for idx, paper_data in enumerate(anonymized_data):
            paper_id = paper_data["paper_id"]
            title = paper_data["title"]
            abstract = paper_data["abstract"]
            pros = paper_data["pros"]
            cons = paper_data["cons"]
            
            # 获取论文全文内容
            paper_content = paper_content_map.get(paper_id, "")
            
            logger.info(f"正在量化论文: {title[:50]}...")
            
            # 构建优缺点文本（只有描述和类别，无审稿人信息）
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
            
            logger.info(f"  输入: {pros_count} 个优点, {cons_count} 个缺点")
            
            # 构建提示词（包含论文全文）
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
                response = self._call_llm(prompt)
                result = safe_json_parse(response, default={
                    "pros_weights": [],
                    "cons_weights": [],
                    "expected_score_breakdown": {}
                })
                
                # 确保权重列表长度匹配
                pros_weights = result.get("pros_weights", [])
                cons_weights = result.get("cons_weights", [])
                
                # 检查并警告数量不匹配
                if len(pros_weights) < pros_count:
                    logger.warning(
                        f"  ⚠ LLM返回的优点权重数量不足: "
                        f"期望 {pros_count}, 实际 {len(pros_weights)}"
                    )
                if len(cons_weights) < cons_count:
                    logger.warning(
                        f"  ⚠ LLM返回的缺点权重数量不足: "
                        f"期望 {cons_count}, 实际 {len(cons_weights)}"
                    )
                
                # 补齐缺失的权重（使用默认值）
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
                
                logger.info(
                    f"  完成: {len(pros_weights)} 优点权重, "
                    f"{len(cons_weights)} 缺点权重"
                )
                
            except Exception as e:
                logger.error(f"量化论文 {paper_id} 失败: {e}")
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
            
            tracker.update(1)
            
            # 请求间隔，防止速率限制
            if idx < len(anonymized_data) - 1:
                logger.info(f"  ⏳ 等待 {Config.REQUEST_DELAY:.1f} 秒...")
                time.sleep(Config.REQUEST_DELAY)
        
        tracker.finish()
        
        # 保存量化结果
        output_file = output_dir / "quantified_weights.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(quantified_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"量化结果已保存到: {output_file}")
        
        return output_file
    
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
    
    # 创建测试匿名数据
    test_anonymized = [
        {
            "paper_id": "test_001",
            "title": "Test Paper",
            "abstract": "Test abstract",
            "pros": [
                {"shuffled_index": 0, "description": "创新方法", "category": "创新性"},
                {"shuffled_index": 1, "description": "实验充分", "category": "实验充分性"}
            ],
            "cons": [
                {"shuffled_index": 0, "description": "写作问题", "category": "写作质量"}
    ]
        }
    ]
    
    # 创建测试论文
    test_paper = Paper(
        paper_id="test_001",
        title="Test Paper",
        abstract="Test abstract",
        paper_content="This is the paper content..."
    )
    
    # 保存测试文件
    test_file = Path("./test_anonymized.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_anonymized, f, ensure_ascii=False, indent=2)
    
    # 测试量化
    quantifier = LLMQuantifier()
    
    try:
        Config.validate()
        output_file = quantifier.quantify_anonymized_file(
            test_file, 
            [test_paper]
        )
        
        print(f"\n✓ 量化器测试完成！")
        print(f"输出文件: {output_file}")
        
    except ValueError as e:
        print(f"\n⚠ 需要配置API密钥才能测试: {e}")
    finally:
        # 清理
        if test_file.exists():
            test_file.unlink()
