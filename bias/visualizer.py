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

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# Standard font settings for plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Solve minus sign issue


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = output_dir or Config.OUTPUT_DIR / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Visualizer initialized. Output directory: {self.output_dir}")
    
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
        Plot bias distribution (Histogram + KDE)
        
        Args:
            results: List of bias analysis results
            filename: Output filename
        """
        # Collect all biases
        all_biases = []
        for result in results:
            all_biases.extend(result.biases)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram and KDE
        sns.histplot(all_biases, kde=True, bins=30, ax=ax)
        
        # Add mean line
        mean_bias = np.mean(all_biases)
        ax.axvline(mean_bias, color='red', linestyle='--', 
                   label=f'Mean: {mean_bias:+.2f}', linewidth=2)
        ax.axvline(0, color='green', linestyle='-', 
                   label='Zero Bias Line', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Bias (Actual Score - Expected Score)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Review Bias Distribution', fontsize=14, fontweight='bold')
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
        Plot Expected Score vs Actual Score scatter plot.
        For large datasets, uses transparency and 2D density to handle overlap.
        
        Args:
            results: List of bias analysis results
            filename: Output filename
        """
        # Collect data
        expected_scores = []
        actual_scores = []
        biases = []
        
        for result in results:
            expected_scores.extend(result.expected_scores)
            actual_scores.extend(result.actual_scores)
            biases.extend(result.biases)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Adjust alpha and markersize for large datasets
        alpha = 0.6 if len(expected_scores) < 100 else 0.3
        s = 100 if len(expected_scores) < 100 else 40
        
        # Color by bias magnitude
        scatter = ax.scatter(expected_scores, actual_scores, 
                           c=biases, cmap='RdBu_r', 
                           alpha=alpha, s=s, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line (perfect match)
        min_score = min(min(expected_scores), min(actual_scores))
        max_score = max(max(expected_scores), max(actual_scores))
        ax.plot([min_score, max_score], [min_score, max_score], 
                'k--', label='Perfect Match Line', linewidth=2, alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Bias', fontsize=12)
        
        # Calculate correlation
        correlation = np.corrcoef(expected_scores, actual_scores)[0, 1]
        
        ax.set_xlabel('Expected Score (LLM Prediction)', fontsize=12)
        ax.set_ylabel('Actual Score (Human Review)', fontsize=12)
        ax.set_title(
            f'Expected Score vs Actual Score (N={len(expected_scores)}, r={correlation:.3f})', 
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        self._save_figure(filename)
        plt.close()

    def plot_bias_by_reviewer_count(
        self,
        results: List[BiasAnalysisResult],
        filename: str = "bias_by_reviewer_count.png"
    ):
        """
        Plot how bias varies with the number of reviews a paper received.
        
        Args:
            results: List of bias analysis results
            filename: Output filename
        """
        counts = [r.num_reviews for r in results]
        biases = [r.bias_mean for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(x=counts, y=biases, ax=ax, palette="Set3")
        sns.stripplot(x=counts, y=biases, ax=ax, color='black', alpha=0.3, size=3)
        
        ax.axhline(0, color='red', linestyle='--', alpha=0.6)
        ax.set_xlabel('Number of Reviews', fontsize=12)
        ax.set_ylabel('Average Bias', fontsize=12)
        ax.set_title('Average Bias by Reviewer Count', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        self._save_figure(filename)
        plt.close()
    
    def plot_bias_by_paper(
        self,
        results: List[BiasAnalysisResult],
        top_n: int = 20,
        filename: str = "bias_by_paper.png"
    ):
        """
        Plot bias distribution across papers.
        For small datasets, shows individual bars.
        For large datasets, shows top/bottom outliers and a general distribution.
        
        Args:
            results: List of bias analysis results
            top_n: Number of top/bottom papers to show as outliers
            filename: Output filename
        """
        if len(results) <= 50:
            # Original behavior for small datasets
            sorted_results = sorted(results, key=lambda x: abs(x.bias_mean), reverse=True)[:top_n]
            paper_labels = [f"{r.paper_title[:30]}..." if len(r.paper_title) > 30 else r.paper_title for r in sorted_results]
            bias_means = [r.bias_mean for r in sorted_results]
            colors = ['red' if b > 0 else 'blue' for b in bias_means]
            
            fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_results) * 0.4)))
            y_pos = np.arange(len(paper_labels))
            ax.barh(y_pos, bias_means, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(paper_labels, fontsize=9)
            ax.set_title(f'Average Bias by Paper (Top {top_n})', fontsize=14, fontweight='bold')
        else:
            # Summary view for large datasets
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1.5]})
            
            # 1. Distribution of average biases
            all_means = [r.bias_mean for r in results]
            sns.histplot(all_means, kde=True, ax=ax1, color='purple')
            ax1.axvline(0, color='black', linestyle='-')
            ax1.set_title('Distribution of Average Bias across All Papers', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Mean Bias')
            
            # 2. Top and Bottom Outliers
            top_positive = sorted(results, key=lambda x: x.bias_mean, reverse=True)[:top_n//2]
            top_negative = sorted(results, key=lambda x: x.bias_mean)[:top_n//2]
            outliers = top_positive + top_negative
            outliers = sorted(outliers, key=lambda x: x.bias_mean)
            
            paper_labels = [f"{r.paper_title[:30]}..." if len(r.paper_title) > 30 else r.paper_title for r in outliers]
            bias_means = [r.bias_mean for r in outliers]
            colors = ['red' if b > 0 else 'blue' for b in bias_means]
            
            y_pos = np.arange(len(outliers))
            ax2.barh(y_pos, bias_means, color=colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(paper_labels, fontsize=9)
            ax2.set_title(f'Extreme Bias Cases (Top {top_n//2} Pos/Neg Outliers)', fontsize=14, fontweight='bold')
            ax2.axvline(0, color='black', linestyle='-')

        plt.tight_layout()
        self._save_figure(filename)
        plt.close()
    
    def plot_consistency_comparison(
        self,
        results: List[BiasAnalysisResult],
        filename: str = "consistency_comparison.png"
    ):
        """
        Plot consistency comparison.
        For large datasets, use a scatter plot of Expected vs Actual Std.
        
        Args:
            results: List of bias analysis results
            filename: Output filename
        """
        expected_stds = [r.expected_std for r in results]
        actual_stds = [r.actual_std for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if len(results) <= 50:
            # Original bar chart for small datasets
            data = [{'paper': r.paper_title[:30] + "...", 'Expected Std': r.expected_std, 'Actual Std': r.actual_std} for r in results[:20]]
            df = pd.DataFrame(data)
            x = np.arange(len(df))
            width = 0.35
            ax.bar(x - width/2, df['Expected Std'], width, label='Expected Score Std', alpha=0.8)
            ax.bar(x + width/2, df['Actual Std'], width, label='Actual Score Std', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(df['paper'], rotation=45, ha='right', fontsize=8)
            ax.set_title('Expected vs Actual Score Consistency (Sample)', fontsize=14, fontweight='bold')
        else:
            # Scatter plot for large datasets
            scatter = ax.scatter(expected_stds, actual_stds, alpha=0.5, c='blue', edgecolors='white')
            
            # Add identity line
            max_std = max(max(expected_stds), max(actual_stds))
            ax.plot([0, max_std], [0, max_std], 'r--', label='Equal Consistency')
            
            # Add trend line
            z = np.polyfit(expected_stds, actual_stds, 1)
            p = np.poly1d(z)
            ax.plot(np.unique(expected_stds), p(np.unique(expected_stds)), "g-", alpha=0.8, label='Trend Line')
            
            ax.set_title(f'Consistency Comparison (N={len(results)} Papers)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Expected Score Std (LLM Consistency)', fontsize=12)
            ax.set_ylabel('Actual Score Std (Human Consistency)', fontsize=12)
            ax.set_aspect('equal')
            
            # Add counts for which side is more consistent
            more_consistent_human = sum(1 for e, a in zip(expected_stds, actual_stds) if a < e)
            more_consistent_llm = sum(1 for e, a in zip(expected_stds, actual_stds) if e < a)
            ax.text(0.05, 0.95, f'Humans more consistent: {more_consistent_human}\nLLM more consistent: {more_consistent_llm}', 
                    transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        self._save_figure(filename)
        plt.close()
    
    def plot_heatmap(
        self,
        results: List[BiasAnalysisResult],
        filename: str = "bias_heatmap.png"
    ):
        """
        Plot bias heatmap.
        For large datasets, show a representative sample or the most biased ones.
        
        Args:
            results: List of bias analysis results
            filename: Output filename
        """
        # Select papers to show
        if len(results) <= 30:
            display_results = results
            title_suffix = ""
        else:
            # Mix of most biased and random ones
            top_biased = sorted(results, key=lambda x: abs(x.bias_mean), reverse=True)[:15]
            random_sample = np.random.choice([r for r in results if r not in top_biased], 
                                            size=min(15, len(results)-len(top_biased)), 
                                            replace=False).tolist()
            display_results = sorted(top_biased + random_sample, key=lambda x: x.bias_mean, reverse=True)
            title_suffix = " (Selected Sample: Outliers + Random)"
            
        max_reviews = max(r.num_reviews for r in display_results)
        bias_matrix = []
        paper_labels = []
        
        for result in display_results:
            row = result.biases + [np.nan] * (max_reviews - len(result.biases))
            bias_matrix.append(row)
            label = result.paper_title[:40] + "..." if len(result.paper_title) > 40 else result.paper_title
            paper_labels.append(label)
        
        bias_matrix = np.array(bias_matrix)
        
        fig, ax = plt.subplots(figsize=(max(10, max_reviews * 1.5), 
                                       max(8, len(paper_labels) * 0.3)))
        
        sns.heatmap(bias_matrix, cmap='RdBu_r', center=0,
                   yticklabels=paper_labels,
                   xticklabels=[f'R{i+1}' for i in range(max_reviews)],
                   annot=True if len(display_results) <= 30 else False, 
                   fmt='.2f', cbar_kws={'label': 'Bias'}, ax=ax, linewidths=0.5)
        
        ax.set_xlabel('Reviewer Index', fontsize=12)
        ax.set_ylabel('Paper', fontsize=12)
        ax.set_title(f'Review Bias Heatmap{title_suffix}', fontsize=14, fontweight='bold')
        
        self._save_figure(filename)
        plt.close()
    
    def plot_box_comparison(
        self,
        results: List[BiasAnalysisResult],
        filename: str = "score_boxplot.png"
    ):
        """
        Plot boxplot comparison of Expected and Actual scores
        
        Args:
            results: List of bias analysis results
            filename: Output filename
        """
        # Prepare data
        data = []
        for result in results:
            for exp, act in zip(result.expected_scores, result.actual_scores):
                data.append({'Score Type': 'Expected Score', 'Score': exp})
                data.append({'Score Type': 'Actual Score', 'Score': act})
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(x='Score Type', y='Score', data=df, ax=ax)
        sns.swarmplot(x='Score Type', y='Score', data=df, 
                     color='black', alpha=0.3, size=3, ax=ax)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('', fontsize=12)
        ax.set_title('Expected vs Actual Score Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        self._save_figure(filename)
        plt.close()
    
    def plot_category_analysis(
        self,
        papers: List[Paper],
        filename: str = "category_analysis.png"
    ):
        """
        Plot statistics of Pro/Con categories
        
        Args:
            papers: List of papers
            filename: Output filename
        """
        # Count categories
        category_pros = {}
        category_cons = {}
        
        for paper in papers:
            for review in paper.reviews:
                for pro in review.pros:
                    cat = pro.get('category', 'Others')
                    category_pros[cat] = category_pros.get(cat, 0) + 1
                
                for con in review.cons:
                    cat = con.get('category', 'Others')
                    category_cons[cat] = category_cons.get(cat, 0) + 1
        
        # Prepare data
        categories = list(set(list(category_pros.keys()) + list(category_cons.keys())))
        pros_counts = [category_pros.get(cat, 0) for cat in categories]
        cons_counts = [-category_cons.get(cat, 0) for cat in categories]  # Negative for symmetric display
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(categories) * 0.4)))
        
        y_pos = np.arange(len(categories))
        
        # Plot symmetric bar chart
        ax.barh(y_pos, pros_counts, label='Pros', color='green', alpha=0.7)
        ax.barh(y_pos, cons_counts, label='Cons', color='red', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=10)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_title('Distribution of Pro/Con Categories', fontsize=14, fontweight='bold')
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
        Plot relationship between bias and actual score
        
        Args:
            results: List of bias analysis results
            filename: Output filename
        """
        # Collect data
        actual_scores = []
        biases = []
        
        for result in results:
            actual_scores.extend(result.actual_scores)
            biases.extend(result.biases)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(actual_scores, biases, alpha=0.5, s=50)
        
        # Add regression line
        z = np.polyfit(actual_scores, biases, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(actual_scores), max(actual_scores), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, 
                label=f'Regression: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # Add zero bias line
        ax.axhline(0, color='green', linestyle='-', 
                  label='Zero Bias Line', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Actual Score', fontsize=12)
        ax.set_ylabel('Bias', fontsize=12)
        ax.set_title('Bias vs Actual Score', fontsize=14, fontweight='bold')
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
        Generate all visualization plots
        
        Args:
            papers: List of papers
            results: List of bias analysis results
        """
        logger.info("Generating all visualization plots...")
        
        plots = [
            (self.plot_bias_distribution, (results,), "Bias Distribution"),
            (self.plot_score_comparison, (results,), "Score Comparison"),
            (self.plot_bias_by_paper, (results,), "Bias by Paper"),
            (self.plot_consistency_comparison, (results,), "Consistency Comparison"),
            (self.plot_heatmap, (results,), "Bias Heatmap"),
            (self.plot_box_comparison, (results,), "Score Boxplot"),
            (self.plot_category_analysis, (papers,), "Category Analysis"),
            (self.plot_bias_vs_score, (results,), "Bias vs Score"),
            (self.plot_bias_by_reviewer_count, (results,), "Bias by Reviewer Count"),
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




