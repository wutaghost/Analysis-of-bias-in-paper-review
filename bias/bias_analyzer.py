"""
偏差分析模块
计算和分析审稿偏差，包括统计检验和相关性分析
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

from config import Config
from data_loader import Paper, Review
from utils import logger, calculate_statistics


@dataclass
class BiasAnalysisResult:
    """偏差分析结果"""
    paper_id: str
    paper_title: str
    num_reviews: int
    
    # 实际分数统计
    actual_scores: List[float]
    actual_mean: float
    actual_std: float
    
    # 期望分数统计
    expected_scores: List[float]
    expected_mean: float
    expected_std: float
    
    # 偏差统计
    biases: List[float]
    bias_mean: float
    bias_std: float
    bias_range: Tuple[float, float]
    
    # 一致性指标
    expected_consistency: float  # 期望分数的一致性（低标准差=高一致性）
    actual_consistency: float    # 实际分数的一致性
    consistency_ratio: float     # 一致性比率（期望/实际）
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "num_reviews": self.num_reviews,
            "actual_mean": self.actual_mean,
            "actual_std": self.actual_std,
            "expected_mean": self.expected_mean,
            "expected_std": self.expected_std,
            "bias_mean": self.bias_mean,
            "bias_std": self.bias_std,
            "bias_min": self.bias_range[0],
            "bias_max": self.bias_range[1],
            "expected_consistency": self.expected_consistency,
            "actual_consistency": self.actual_consistency,
            "consistency_ratio": self.consistency_ratio,
        }


class BiasAnalyzer:
    """偏差分析器"""
    
    def __init__(self):
        logger.info("偏差分析器已初始化")
    
    def analyze_paper(self, paper: Paper) -> BiasAnalysisResult:
        """
        分析单篇论文的审稿偏差
        
        Args:
            paper: 论文对象（必须已完成权重量化）
            
        Returns:
            偏差分析结果
        """
        # 检查是否已完成量化
        if not paper.reviews:
            raise ValueError(f"论文 {paper.paper_id} 没有审稿记录")
        
        if paper.reviews[0].expected_score is None:
            raise ValueError(
                f"论文 {paper.paper_id} 的审稿记录未完成权重量化"
            )
        
        # 收集数据
        actual_scores = [r.actual_score for r in paper.reviews]
        expected_scores = [r.expected_score for r in paper.reviews]
        biases = [r.bias for r in paper.reviews]
        
        # 计算统计指标
        actual_mean = np.mean(actual_scores)
        actual_std = np.std(actual_scores)
        expected_mean = np.mean(expected_scores)
        expected_std = np.std(expected_scores)
        bias_mean = np.mean(biases)
        bias_std = np.std(biases)
        
        # 一致性指标（使用变异系数的倒数作为一致性度量）
        # 变异系数 = std / mean，越小越一致
        # 一致性 = 1 / (1 + CV)，范围[0, 1]，越大越一致
        expected_cv = expected_std / abs(expected_mean) if expected_mean != 0 else float('inf')
        actual_cv = actual_std / abs(actual_mean) if actual_mean != 0 else float('inf')
        
        expected_consistency = 1 / (1 + expected_cv)
        actual_consistency = 1 / (1 + actual_cv)
        
        # 一致性比率：期望分数的一致性 / 实际分数的一致性
        # > 1 表示基于相同优缺点的期望分数更一致
        consistency_ratio = expected_consistency / actual_consistency if actual_consistency > 0 else 0
        
        result = BiasAnalysisResult(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            num_reviews=len(paper.reviews),
            actual_scores=actual_scores,
            actual_mean=actual_mean,
            actual_std=actual_std,
            expected_scores=expected_scores,
            expected_mean=expected_mean,
            expected_std=expected_std,
            biases=biases,
            bias_mean=bias_mean,
            bias_std=bias_std,
            bias_range=(min(biases), max(biases)),
            expected_consistency=expected_consistency,
            actual_consistency=actual_consistency,
            consistency_ratio=consistency_ratio,
        )
        
        # 更新paper对象的统计信息
        paper.avg_actual_score = actual_mean
        paper.std_actual_score = actual_std
        paper.avg_expected_score = expected_mean
        paper.std_expected_score = expected_std
        paper.avg_bias = bias_mean
        
        logger.debug(
            f"论文 {paper.title}: "
            f"实际={actual_mean:.2f}±{actual_std:.2f}, "
            f"期望={expected_mean:.2f}±{expected_std:.2f}, "
            f"偏差={bias_mean:+.2f}±{bias_std:.2f}"
        )
        
        return result
    
    def analyze_papers(self, papers: List[Paper]) -> List[BiasAnalysisResult]:
        """
        批量分析多篇论文的偏差
        
        Args:
            papers: 论文列表
            
        Returns:
            偏差分析结果列表
        """
        logger.info(f"开始分析 {len(papers)} 篇论文的偏差...")
        
        results = []
        for paper in papers:
            try:
                result = self.analyze_paper(paper)
                results.append(result)
            except Exception as e:
                logger.error(f"分析论文 {paper.paper_id} 时出错: {e}")
        
        logger.info(f"完成 {len(results)} 篇论文的偏差分析")
        
        return results
    
    def global_statistics(
        self, 
        results: List[BiasAnalysisResult]
    ) -> Dict[str, Any]:
        """
        计算全局统计指标
        
        Args:
            results: 偏差分析结果列表
            
        Returns:
            全局统计字典
        """
        # 收集所有数据
        all_biases = []
        all_actual_scores = []
        all_expected_scores = []
        all_consistency_ratios = []
        
        for result in results:
            all_biases.extend(result.biases)
            all_actual_scores.extend(result.actual_scores)
            all_expected_scores.extend(result.expected_scores)
            all_consistency_ratios.append(result.consistency_ratio)
        
        # 计算统计指标
        bias_stats = calculate_statistics(all_biases)
        actual_stats = calculate_statistics(all_actual_scores)
        expected_stats = calculate_statistics(all_expected_scores)
        
        # 相关性分析
        correlation = np.corrcoef(all_expected_scores, all_actual_scores)[0, 1]
        
        # MAE和RMSE
        mae = np.mean(np.abs(all_biases))
        rmse = np.sqrt(np.mean(np.square(all_biases)))
        
        # 配对t检验：检验期望分数和实际分数是否有显著差异
        t_stat, p_value = stats.ttest_rel(all_expected_scores, all_actual_scores)
        
        # 卡方检验：检验偏差是否服从正态分布
        # Shapiro-Wilk检验
        if len(all_biases) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(all_biases)
        else:
            shapiro_stat, shapiro_p = None, None
        
        # 偏差的正负比例
        positive_biases = sum(1 for b in all_biases if b > 0)
        negative_biases = sum(1 for b in all_biases if b < 0)
        zero_biases = len(all_biases) - positive_biases - negative_biases
        
        statistics = {
            "total_papers": len(results),
            "total_reviews": len(all_biases),
            "bias_statistics": {
                **bias_stats,
                "mae": mae,
                "rmse": rmse,
                "positive_count": positive_biases,
                "negative_count": negative_biases,
                "zero_count": zero_biases,
                "positive_ratio": positive_biases / len(all_biases) if all_biases else 0,
            },
            "actual_score_statistics": actual_stats,
            "expected_score_statistics": expected_stats,
            "correlation": {
                "pearson_r": correlation,
                "r_squared": correlation ** 2,
            },
            "hypothesis_tests": {
                "paired_t_test": {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "interpretation": (
                        "期望分数和实际分数有显著差异" if p_value < 0.05 
                        else "期望分数和实际分数无显著差异"
                    )
                },
                "normality_test": {
                    "shapiro_statistic": float(shapiro_stat) if shapiro_stat else None,
                    "p_value": float(shapiro_p) if shapiro_p else None,
                    "normal": shapiro_p > 0.05 if shapiro_p else None,
                } if shapiro_stat else None,
            },
            "consistency_analysis": {
                "avg_consistency_ratio": np.mean(all_consistency_ratios),
                "interpretation": (
                    "基于相同优缺点的期望分数比实际分数更一致" 
                    if np.mean(all_consistency_ratios) > 1 
                    else "实际分数比期望分数更一致（反常）"
                ),
                "papers_with_high_consistency_ratio": sum(
                    1 for r in all_consistency_ratios if r > 1.2
                ),
            }
        }
        
        return statistics
    
    def identify_high_bias_cases(
        self,
        results: List[BiasAnalysisResult],
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        识别高偏差案例
        
        Args:
            results: 偏差分析结果列表
            threshold: 偏差阈值（绝对值）
            
        Returns:
            高偏差案例列表
        """
        high_bias_cases = []
        
        for result in results:
            for i, bias in enumerate(result.biases):
                if abs(bias) >= threshold:
                    high_bias_cases.append({
                        "paper_id": result.paper_id,
                        "paper_title": result.paper_title,
                        "reviewer_index": i,
                        "actual_score": result.actual_scores[i],
                        "expected_score": result.expected_scores[i],
                        "bias": bias,
                        "bias_type": "正偏差(高估)" if bias > 0 else "负偏差(低估)",
                    })
        
        # 按偏差绝对值排序
        high_bias_cases.sort(key=lambda x: abs(x["bias"]), reverse=True)
        
        logger.info(
            f"识别出 {len(high_bias_cases)} 个高偏差案例 "
            f"(阈值: {threshold})"
        )
        
        return high_bias_cases
    
    def compare_papers_by_consistency(
        self,
        results: List[BiasAnalysisResult]
    ) -> pd.DataFrame:
        """
        按一致性比较论文
        
        Args:
            results: 偏差分析结果列表
            
        Returns:
            比较结果的DataFrame
        """
        data = []
        
        for result in results:
            data.append({
                "论文ID": result.paper_id,
                "论文标题": result.paper_title[:50] + "...",  # 截断标题
                "审稿数": result.num_reviews,
                "实际分数均值": result.actual_mean,
                "实际分数标准差": result.actual_std,
                "期望分数均值": result.expected_mean,
                "期望分数标准差": result.expected_std,
                "偏差均值": result.bias_mean,
                "一致性比率": result.consistency_ratio,
                "偏差": "高" if abs(result.bias_mean) > 1.5 else "中" if abs(result.bias_mean) > 0.5 else "低",
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("一致性比率", ascending=False)
        
        return df
    
    def generate_summary_report(
        self,
        results: List[BiasAnalysisResult]
    ) -> str:
        """
        生成文本摘要报告
        
        Args:
            results: 偏差分析结果列表
            
        Returns:
            摘要报告文本
        """
        stats = self.global_statistics(results)
        high_bias = self.identify_high_bias_cases(results, threshold=2.0)
        
        report = []
        report.append("=" * 70)
        report.append("审稿偏差分析报告")
        report.append("=" * 70)
        report.append("")
        
        # 基本统计
        report.append("一、基本统计")
        report.append("-" * 70)
        report.append(f"分析论文数: {stats['total_papers']}")
        report.append(f"审稿记录数: {stats['total_reviews']}")
        report.append("")
        
        # 偏差分析
        report.append("二、偏差分析")
        report.append("-" * 70)
        bias_stats = stats['bias_statistics']
        report.append(f"平均偏差: {bias_stats['mean']:+.3f}")
        report.append(f"偏差标准差: {bias_stats['std']:.3f}")
        report.append(f"偏差中位数: {bias_stats['median']:+.3f}")
        report.append(f"偏差范围: [{bias_stats['min']:+.3f}, {bias_stats['max']:+.3f}]")
        report.append(f"平均绝对误差(MAE): {bias_stats['mae']:.3f}")
        report.append(f"均方根误差(RMSE): {bias_stats['rmse']:.3f}")
        report.append("")
        report.append(f"正偏差(高估)数量: {bias_stats['positive_count']} ({bias_stats['positive_ratio']*100:.1f}%)")
        report.append(f"负偏差(低估)数量: {bias_stats['negative_count']}")
        report.append(f"无偏差数量: {bias_stats['zero_count']}")
        report.append("")
        
        # 相关性分析
        report.append("三、相关性分析")
        report.append("-" * 70)
        corr = stats['correlation']
        report.append(f"Pearson相关系数: {corr['pearson_r']:.3f}")
        report.append(f"决定系数(R²): {corr['r_squared']:.3f}")
        report.append("")
        
        # 假设检验
        report.append("四、假设检验")
        report.append("-" * 70)
        t_test = stats['hypothesis_tests']['paired_t_test']
        report.append(f"配对t检验:")
        report.append(f"  t统计量: {t_test['t_statistic']:.3f}")
        report.append(f"  p值: {t_test['p_value']:.4f}")
        report.append(f"  结论: {t_test['interpretation']}")
        report.append("")
        
        # 一致性分析
        report.append("五、一致性分析")
        report.append("-" * 70)
        cons = stats['consistency_analysis']
        report.append(f"平均一致性比率: {cons['avg_consistency_ratio']:.3f}")
        report.append(f"结论: {cons['interpretation']}")
        report.append(f"高一致性比率论文数(>1.2): {cons['papers_with_high_consistency_ratio']}")
        report.append("")
        
        # 高偏差案例
        report.append("六、典型高偏差案例（Top 10）")
        report.append("-" * 70)
        for i, case in enumerate(high_bias[:10], 1):
            report.append(
                f"{i}. {case['paper_title'][:40]}..."
            )
            report.append(
                f"   实际分数={case['actual_score']:.1f}, "
                f"期望分数={case['expected_score']:.1f}, "
                f"偏差={case['bias']:+.2f} ({case['bias_type']})"
            )
            report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def display_summary(self, results: List[BiasAnalysisResult]):
        """显示摘要信息"""
        report = self.generate_summary_report(results)
        print(report)


if __name__ == "__main__":
    # 测试偏差分析器
    from data_loader import Paper, Review
    
    # 创建测试数据
    test_paper = Paper(
        paper_id="test_paper",
        title="Test Paper",
        abstract="Test abstract"
    )
    
    # 创建多个审稿记录
    for i in range(3):
        review = Review(
            reviewer_id=f"reviewer_{i}",
            review_text="Test review",
            actual_score=7.0 + i
        )
        # 模拟已量化的结果
        review.expected_score = 6.5 + i * 0.5
        review.bias = review.actual_score - review.expected_score
        review.pros = []
        review.cons = []
        test_paper.add_review(review)
    
    # 测试分析
    analyzer = BiasAnalyzer()
    result = analyzer.analyze_paper(test_paper)
    
    print("\n偏差分析结果:")
    print(f"实际分数: {result.actual_mean:.2f} ± {result.actual_std:.2f}")
    print(f"期望分数: {result.expected_mean:.2f} ± {result.expected_std:.2f}")
    print(f"平均偏差: {result.bias_mean:+.2f}")
    print(f"一致性比率: {result.consistency_ratio:.2f}")
    
    # 测试全局统计
    results = [result]
    stats = analyzer.global_statistics(results)
    analyzer.display_summary(results)
    
    print("\n✓ 偏差分析器测试完成！")




