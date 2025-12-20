"""
可视化模块
生成各种统计图表展示偏差分析结果
"""

from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from config import Config
from data_loader import Paper
from bias_analyzer import BiasAnalysisResult
from utils import logger

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化可视化工具
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir or Config.OUTPUT_DIR / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"可视化工具已初始化，输出目录: {self.output_dir}")
    
    def _save_figure(self, filename: str, dpi: int = None):
        """保存图表"""
        dpi = dpi or Config.FIGURE_DPI
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"图表已保存: {filepath}")
    
    def plot_bias_distribution(
        self, 
        results: List[BiasAnalysisResult],
        filename: str = "bias_distribution.png"
    ):
        """
        绘制偏差分布图（直方图 + KDE）
        
        Args:
            results: 偏差分析结果列表
            filename: 保存文件名
        """
        # 收集所有偏差
        all_biases = []
        for result in results:
            all_biases.extend(result.biases)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制直方图和KDE
        sns.histplot(all_biases, kde=True, bins=30, ax=ax)
        
        # 添加均值线
        mean_bias = np.mean(all_biases)
        ax.axvline(mean_bias, color='red', linestyle='--', 
                   label=f'均值: {mean_bias:+.2f}', linewidth=2)
        ax.axvline(0, color='green', linestyle='-', 
                   label='零偏差线', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('偏差 (实际分数 - 期望分数)', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('审稿偏差分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        self._save_figure(filename)
        plt.close()
    
    def plot_score_comparison(
        self,
        results: List[BiasAnalysisResult],
        filename: str = "score_comparison.png"
    ):
        """
        绘制期望分数 vs 实际分数散点图
        
        Args:
            results: 偏差分析结果列表
            filename: 保存文件名
        """
        # 收集数据
        expected_scores = []
        actual_scores = []
        biases = []
        
        for result in results:
            expected_scores.extend(result.expected_scores)
            actual_scores.extend(result.actual_scores)
            biases.extend(result.biases)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 根据偏差大小设置颜色
        scatter = ax.scatter(expected_scores, actual_scores, 
                           c=biases, cmap='RdBu_r', 
                           alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        # 添加对角线（完美匹配线）
        min_score = min(min(expected_scores), min(actual_scores))
        max_score = max(max(expected_scores), max(actual_scores))
        ax.plot([min_score, max_score], [min_score, max_score], 
                'k--', label='完美匹配线', linewidth=2, alpha=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('偏差', fontsize=12)
        
        # 计算相关系数
        correlation = np.corrcoef(expected_scores, actual_scores)[0, 1]
        
        ax.set_xlabel('期望分数', fontsize=12)
        ax.set_ylabel('实际分数', fontsize=12)
        ax.set_title(
            f'期望分数 vs 实际分数 (r={correlation:.3f})', 
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        self._save_figure(filename)
        plt.close()
    
    def plot_bias_by_paper(
        self,
        results: List[BiasAnalysisResult],
        top_n: int = 20,
        filename: str = "bias_by_paper.png"
    ):
        """
        绘制每篇论文的平均偏差（柱状图）
        
        Args:
            results: 偏差分析结果列表
            top_n: 显示前N篇（按偏差绝对值排序）
            filename: 保存文件名
        """
        # 按偏差绝对值排序
        sorted_results = sorted(
            results, 
            key=lambda x: abs(x.bias_mean), 
            reverse=True
        )[:top_n]
        
        # 准备数据
        paper_labels = [
            f"{r.paper_title[:30]}..." if len(r.paper_title) > 30 
            else r.paper_title 
            for r in sorted_results
        ]
        bias_means = [r.bias_mean for r in sorted_results]
        colors = ['red' if b > 0 else 'blue' for b in bias_means]
        
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
        
        y_pos = np.arange(len(paper_labels))
        ax.barh(y_pos, bias_means, color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(paper_labels, fontsize=9)
        ax.set_xlabel('平均偏差', fontsize=12)
        ax.set_title(
            f'各论文平均偏差 (Top {top_n})', 
            fontsize=14, fontweight='bold'
        )
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='正偏差（高估）'),
            Patch(facecolor='blue', alpha=0.7, label='负偏差（低估）')
        ]
        ax.legend(handles=legend_elements, fontsize=10)
        
        self._save_figure(filename)
        plt.close()
    
    def plot_consistency_comparison(
        self,
        results: List[BiasAnalysisResult],
        filename: str = "consistency_comparison.png"
    ):
        """
        绘制一致性比较图
        
        Args:
            results: 偏差分析结果列表
            filename: 保存文件名
        """
        # 准备数据
        data = []
        for result in results:
            data.append({
                'paper': result.paper_title[:30] + "...",
                '期望分数标准差': result.expected_std,
                '实际分数标准差': result.actual_std,
            })
        
        df = pd.DataFrame(data)
        
        # 选择前20个论文
        df = df.head(20)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['期望分数标准差'], width, 
               label='期望分数标准差', alpha=0.8)
        ax.bar(x + width/2, df['实际分数标准差'], width,
               label='实际分数标准差', alpha=0.8)
        
        ax.set_xlabel('论文', fontsize=12)
        ax.set_ylabel('标准差（越小越一致）', fontsize=12)
        ax.set_title('期望分数 vs 实际分数的一致性对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['paper'], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        self._save_figure(filename)
        plt.close()
    
    def plot_heatmap(
        self,
        results: List[BiasAnalysisResult],
        filename: str = "bias_heatmap.png"
    ):
        """
        绘制偏差热力图（论文 x 审稿人）
        
        Args:
            results: 偏差分析结果列表
            filename: 保存文件名
        """
        # 准备数据矩阵
        max_reviews = max(r.num_reviews for r in results)
        bias_matrix = []
        paper_labels = []
        
        for result in results[:30]:  # 最多显示30篇论文
            row = result.biases + [np.nan] * (max_reviews - len(result.biases))
            bias_matrix.append(row)
            label = result.paper_title[:40] + "..." if len(result.paper_title) > 40 else result.paper_title
            paper_labels.append(label)
        
        bias_matrix = np.array(bias_matrix)
        
        fig, ax = plt.subplots(figsize=(max(10, max_reviews * 1.5), 
                                       max(8, len(paper_labels) * 0.3)))
        
        # 绘制热力图
        sns.heatmap(bias_matrix, 
                   cmap='RdBu_r', 
                   center=0,
                   yticklabels=paper_labels,
                   xticklabels=[f'R{i+1}' for i in range(max_reviews)],
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': '偏差'},
                   ax=ax,
                   linewidths=0.5)
        
        ax.set_xlabel('审稿人', fontsize=12)
        ax.set_ylabel('论文', fontsize=12)
        ax.set_title('审稿偏差热力图', fontsize=14, fontweight='bold')
        
        self._save_figure(filename)
        plt.close()
    
    def plot_box_comparison(
        self,
        results: List[BiasAnalysisResult],
        filename: str = "score_boxplot.png"
    ):
        """
        绘制期望分数和实际分数的箱线图对比
        
        Args:
            results: 偏差分析结果列表
            filename: 保存文件名
        """
        # 准备数据
        data = []
        for result in results:
            for exp, act in zip(result.expected_scores, result.actual_scores):
                data.append({'分数类型': '期望分数', '分数': exp})
                data.append({'分数类型': '实际分数', '分数': act})
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(x='分数类型', y='分数', data=df, ax=ax)
        sns.swarmplot(x='分数类型', y='分数', data=df, 
                     color='black', alpha=0.3, size=3, ax=ax)
        
        ax.set_ylabel('分数', fontsize=12)
        ax.set_xlabel('', fontsize=12)
        ax.set_title('期望分数 vs 实际分数分布对比', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        self._save_figure(filename)
        plt.close()
    
    def plot_category_analysis(
        self,
        papers: List[Paper],
        filename: str = "category_analysis.png"
    ):
        """
        绘制优缺点类别统计图
        
        Args:
            papers: 论文列表
            filename: 保存文件名
        """
        # 统计各类别
        category_pros = {}
        category_cons = {}
        
        for paper in papers:
            for review in paper.reviews:
                for pro in review.pros:
                    cat = pro.get('category', '其他')
                    category_pros[cat] = category_pros.get(cat, 0) + 1
                
                for con in review.cons:
                    cat = con.get('category', '其他')
                    category_cons[cat] = category_cons.get(cat, 0) + 1
        
        # 准备数据
        categories = list(set(list(category_pros.keys()) + list(category_cons.keys())))
        pros_counts = [category_pros.get(cat, 0) for cat in categories]
        cons_counts = [-category_cons.get(cat, 0) for cat in categories]  # 负值用于对称显示
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(categories) * 0.4)))
        
        y_pos = np.arange(len(categories))
        
        # 绘制对称柱状图
        ax.barh(y_pos, pros_counts, label='优点', color='green', alpha=0.7)
        ax.barh(y_pos, cons_counts, label='缺点', color='red', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=10)
        ax.set_xlabel('数量', fontsize=12)
        ax.set_title('各类别优缺点分布', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        self._save_figure(filename)
        plt.close()
    
    def plot_bias_vs_score(
        self,
        results: List[BiasAnalysisResult],
        filename: str = "bias_vs_actual_score.png"
    ):
        """
        绘制偏差与实际分数的关系
        
        Args:
            results: 偏差分析结果列表
            filename: 保存文件名
        """
        # 收集数据
        actual_scores = []
        biases = []
        
        for result in results:
            actual_scores.extend(result.actual_scores)
            biases.extend(result.biases)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 散点图
        ax.scatter(actual_scores, biases, alpha=0.5, s=50)
        
        # 添加回归线
        z = np.polyfit(actual_scores, biases, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(actual_scores), max(actual_scores), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, 
                label=f'回归线: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # 添加零偏差线
        ax.axhline(0, color='green', linestyle='-', 
                  label='零偏差线', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('实际分数', fontsize=12)
        ax.set_ylabel('偏差', fontsize=12)
        ax.set_title('偏差与实际分数的关系', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        self._save_figure(filename)
        plt.close()
    
    def generate_all_plots(
        self,
        papers: List[Paper],
        results: List[BiasAnalysisResult]
    ):
        """
        生成所有可视化图表
        
        Args:
            papers: 论文列表
            results: 偏差分析结果列表
        """
        logger.info("开始生成所有可视化图表...")
        
        plots = [
            (self.plot_bias_distribution, (results,), "偏差分布图"),
            (self.plot_score_comparison, (results,), "分数对比散点图"),
            (self.plot_bias_by_paper, (results,), "论文偏差柱状图"),
            (self.plot_consistency_comparison, (results,), "一致性对比图"),
            (self.plot_heatmap, (results,), "偏差热力图"),
            (self.plot_box_comparison, (results,), "箱线图对比"),
            (self.plot_category_analysis, (papers,), "类别统计图"),
            (self.plot_bias_vs_score, (results,), "偏差-分数关系图"),
        ]
        
        for plot_func, args, description in plots:
            try:
                logger.info(f"正在生成: {description}")
                plot_func(*args)
            except Exception as e:
                logger.error(f"生成{description}失败: {e}")
        
        logger.info(f"所有图表已保存到: {self.output_dir}")


if __name__ == "__main__":
    # 测试可视化工具
    from data_loader import Paper, Review
    from bias_analyzer import BiasAnalyzer
    
    # 创建测试数据
    test_papers = []
    for i in range(5):
        paper = Paper(
            paper_id=f"paper_{i}",
            title=f"Test Paper {i}",
            abstract="Test abstract"
        )
        
        for j in range(3):
            review = Review(
                reviewer_id=f"reviewer_{j}",
                review_text="Test",
                actual_score=np.random.uniform(5, 9)
            )
            review.expected_score = np.random.uniform(5, 9)
            review.bias = review.actual_score - review.expected_score
            review.pros = [{"description": "test", "category": "创新性 (Novelty/Originality)"}]
            review.cons = [{"description": "test", "category": "写作质量 (Writing Quality)"}]
            paper.add_review(review)
        
        test_papers.append(paper)
    
    # 分析
    analyzer = BiasAnalyzer()
    results = analyzer.analyze_papers(test_papers)
    
    # 可视化
    visualizer = Visualizer()
    visualizer.generate_all_plots(test_papers, results)
    
    print("\n✓ 可视化工具测试完成！")
    print(f"图表保存在: {visualizer.output_dir}")




