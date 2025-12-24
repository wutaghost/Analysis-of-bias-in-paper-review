"""
数据加载和预处理模块
支持JSON和CSV格式的论文审稿数据加载
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict

from utils import logger, extract_text_from_pdf


@dataclass
class Review:
    """审稿记录"""
    reviewer_id: str
    review_text: str
    actual_score: float
    
    # 特征提取结果（后续填充）
    pros: List[Dict[str, str]] = field(default_factory=list)
    cons: List[Dict[str, str]] = field(default_factory=list)
    
    # 权重量化结果（后续填充）
    pros_weights: List[Dict[str, Any]] = field(default_factory=list)
    cons_weights: List[Dict[str, Any]] = field(default_factory=list)
    
    # 分数计算结果（后续填充）
    expected_score: Optional[float] = None
    bias: Optional[float] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class Paper:
    """论文记录"""
    paper_id: str
    title: str
    abstract: str
    paper_content: Optional[str] = None
    source_dir: Optional[Path] = None  # 记录源目录以便延迟加载 PDF
    reviews: List[Review] = field(default_factory=list)
    
    # 统计信息（后续填充）
    avg_actual_score: Optional[float] = None
    std_actual_score: Optional[float] = None
    avg_expected_score: Optional[float] = None
    std_expected_score: Optional[float] = None
    avg_bias: Optional[float] = None
    
    def add_review(self, review: Review):
        """添加审稿记录"""
        self.reviews.append(review)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['reviews'] = [r.to_dict() for r in self.reviews]
        return data


class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.papers: List[Paper] = []
        self.data_format: Optional[str] = None
    
    def load_from_json(self, file_path: Union[str, Path]) -> List[Paper]:
        """
        从JSON文件加载数据
        
        预期格式:
        [
          {
            "paper_id": "paper_001",
            "title": "论文标题",
            "abstract": "论文摘要",
            "paper_content": "论文全文（可选）",
            "reviews": [
              {
                "reviewer_id": "reviewer_1",
                "review_text": "审稿意见",
                "actual_score": 7
              }
            ]
          }
        ]
        """
        file_path = Path(file_path)
        logger.info(f"从JSON文件加载数据: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON数据应该是一个列表")
            
            papers = []
            for item in data:
                paper = Paper(
                    paper_id=item.get('paper_id', ''),
                    title=item.get('title', ''),
                    abstract=item.get('abstract', ''),
                    paper_content=item.get('paper_content'),
                )
                
                # 加载审稿记录
                for review_data in item.get('reviews', []):
                    review = Review(
                        reviewer_id=review_data.get('reviewer_id', ''),
                        review_text=review_data.get('review_text', ''),
                        actual_score=float(review_data.get('actual_score', 0)),
                    )
                    paper.add_review(review)
                
                papers.append(paper)
            
            self.papers = papers
            self.data_format = 'json'
            
            logger.info(
                f"成功加载 {len(papers)} 篇论文，"
                f"共 {sum(len(p.reviews) for p in papers)} 条审稿记录"
            )
            
            return papers
            
        except Exception as e:
            logger.error(f"加载JSON数据失败: {e}")
            raise
    
    def load_from_csv(self, file_path: Union[str, Path]) -> List[Paper]:
        """
        从CSV文件加载数据
        
        预期列:
        - paper_id: 论文ID
        - title: 论文标题
        - abstract: 论文摘要
        - paper_content: 论文全文（可选）
        - reviewer_id: 审稿人ID
        - review_text: 审稿意见
        - actual_score: 实际打分
        """
        file_path = Path(file_path)
        logger.info(f"从CSV文件加载数据: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # 验证必需的列
            required_cols = ['paper_id', 'title', 'abstract', 
                           'reviewer_id', 'review_text', 'actual_score']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"CSV缺少必需的列: {missing_cols}")
            
            # 按paper_id分组
            papers_dict: Dict[str, Paper] = {}
            
            for _, row in df.iterrows():
                paper_id = str(row['paper_id'])
                
                # 创建或获取Paper对象
                if paper_id not in papers_dict:
                    papers_dict[paper_id] = Paper(
                        paper_id=paper_id,
                        title=str(row['title']),
                        abstract=str(row['abstract']),
                        paper_content=str(row.get('paper_content', '')) 
                                     if pd.notna(row.get('paper_content')) else None,
                    )
                
                # 添加审稿记录
                review = Review(
                    reviewer_id=str(row['reviewer_id']),
                    review_text=str(row['review_text']),
                    actual_score=float(row['actual_score']),
                )
                papers_dict[paper_id].add_review(review)
            
            self.papers = list(papers_dict.values())
            self.data_format = 'csv'
            
            logger.info(
                f"成功加载 {len(self.papers)} 篇论文，"
                f"共 {sum(len(p.reviews) for p in self.papers)} 条审稿记录"
            )
            
            return self.papers
            
        except Exception as e:
            logger.error(f"加载CSV数据失败: {e}")
            raise
    
    def load_from_openreview_json(self, directory: Union[str, Path]) -> List[Paper]:
        """
        从OpenReview导出的JSON文件批量加载数据
        
        ICLR 2025格式:
        - 顶层键: 'paper' (论文信息) 和 'reviews' (审稿列表)
        - 分数字段: 'rating_or_recommendation'
        - 文本字段: 'body', 'strengths', 'weaknesses', 'questions'
        """
        directory = Path(directory)
        logger.info(f"从OpenReview目录加载数据: {directory}")
        
        papers = []
        json_files = list(directory.rglob("*_openreview.json"))
        
        logger.info(f"找到 {len(json_files)} 个OpenReview JSON文件")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取论文信息（支持两种格式）
                if 'paper' in data:
                    # ICLR 2025格式：有paper字段
                    paper_info = data['paper']
                    paper_id = paper_info.get('id', json_file.stem)
                    title = paper_info.get('title', '')
                    abstract = paper_info.get('abstract', '')
                else:
                    # 旧格式：直接在顶层
                    paper_id = data.get('id', json_file.stem)
                    title = data.get('title', '')
                    abstract = data.get('abstract', '')
                
                paper = Paper(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                )
                
                # 提取审稿记录
                reviews_data = data.get('reviews', [])
                if isinstance(reviews_data, dict):
                    reviews_data = list(reviews_data.values())
                
                for idx, review_data in enumerate(reviews_data):
                    # 提取审稿人ID
                    reviewer_id = review_data.get('actor', 
                                  review_data.get('signatures', 
                                  review_data.get('id', f"reviewer_{idx}")))
                    if isinstance(reviewer_id, list):
                        reviewer_id = reviewer_id[0] if reviewer_id else f"reviewer_{idx}"
                    
                    # 拼接审稿文本 - ICLR 2025 标准顺序
                    review_text_parts = []
                    
                    # 按优先级顺序提取文本字段
                    if 'body' in review_data and review_data['body']:
                        review_text_parts.append(f"Review Body:\n{review_data['body']}")
                    
                    if 'strengths' in review_data and review_data['strengths']:
                        review_text_parts.append(f"Strengths:\n{review_data['strengths']}")
                    
                    if 'weaknesses' in review_data and review_data['weaknesses']:
                        review_text_parts.append(f"Weaknesses:\n{review_data['weaknesses']}")
                    
                    if 'questions' in review_data and review_data['questions']:
                        review_text_parts.append(f"Questions:\n{review_data['questions']}")
                    
                    # 如果以上字段都没有，尝试其他字段
                    if not review_text_parts:
                        for field in ['summary', 'review', 'comments']:
                            if field in review_data and review_data[field]:
                                review_text_parts.append(str(review_data[field]))
                    
                    review_text = "\n\n".join(review_text_parts)
                    
                    # 如果没有任何文本，跳过这条审稿
                    if not review_text.strip():
                        logger.warning(
                            f"论文 {title[:50]} 的审稿 {reviewer_id} 没有文本内容"
                        )
                        continue
                    
                    # 提取分数 - ICLR 2025 使用 rating_or_recommendation
                    actual_score = None
                    for score_field in ['rating_or_recommendation', 'rating', 'score', 'recommendation', 'overall_assessment']:
                        if score_field in review_data and review_data[score_field] is not None:
                            score_value = review_data[score_field]
                            # 处理 "8: Accept" 或 "Accept (8)" 这样的格式
                            if isinstance(score_value, str):
                                # 尝试提取数字
                                import re
                                numbers = re.findall(r'\d+\.?\d*', score_value)
                                if numbers:
                                    score_value = numbers[0]
                            try:
                                actual_score = float(score_value)
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    if actual_score is None:
                        logger.warning(
                            f"论文 {title[:50]}... 的审稿 {reviewer_id} 没有找到有效分数"
                        )
                        continue
                    
                    # 分数合理性检查（ICLR通常是1-10分）
                    if actual_score < 1 or actual_score > 10:
                        logger.warning(
                            f"论文 {title[:50]}... 的审稿 {reviewer_id} 分数异常: {actual_score}"
                        )
                        # 尝试归一化
                        if actual_score > 10:
                            actual_score = min(actual_score / 10, 10.0)
                    
                    review = Review(
                        reviewer_id=reviewer_id,
                        review_text=review_text,
                        actual_score=actual_score,
                    )
                    paper.add_review(review)
                
                if paper.reviews:  # 只添加有有效审稿记录的论文
                    # 仅记录源目录，不再此处直接提取 PDF 内容（优化为按需提取）
                    paper.source_dir = json_file.parent
                    
                    papers.append(paper)
                    logger.debug(
                        f"✓ 加载论文数据: {title[:50]}{'...' if len(title) > 50 else ''} "
                        f"({len(paper.reviews)} 条审稿)"
                    )
                else:
                    logger.debug(f"✗ 跳过论文: {title[:50]}... (无有效审稿)")
                
            except Exception as e:
                logger.warning(f"加载文件 {json_file.name} 失败: {e}")
                continue
        
        self.papers = papers
        self.data_format = 'openreview_json'
        
        logger.info(
            f"成功加载 {len(papers)} 篇论文，"
            f"共 {sum(len(p.reviews) for p in papers)} 条审稿记录"
        )
        
        return papers
    
    def _load_pdf_content(self, paper: Paper, directory: Path):
        """尝试查找并加载论文的 PDF 内容"""
        # 1. 尝试基于 paper_id 查找
        pdf_candidates = [
            directory / f"{paper.paper_id}.pdf",
            directory / f"{paper.paper_id}_openreview.pdf",
            directory / f"{paper.paper_id.replace('/', '_')}.pdf"
        ]
        
        for pdf_path in pdf_candidates:
            if pdf_path.exists():
                logger.info(f"  找到 PDF 文件 (ID匹配): {pdf_path.name}")
                paper.paper_content = extract_text_from_pdf(pdf_path)
                if paper.paper_content:
                    logger.info(f"    成功提取 {len(paper.paper_content)} 字符")
                    return
        
        # 2. 如果没找到，尝试在目录下找任何 PDF 文件
        all_pdfs = list(directory.glob("*.pdf"))
        if all_pdfs:
            # 优先选择最像论文的一个（比如不是 openreview 结尾的，或者名字最长的）
            # 这里简单取第一个
            pdf_path = all_pdfs[0]
            logger.info(f"  找到 PDF 文件 (目录匹配): {pdf_path.name}")
            paper.paper_content = extract_text_from_pdf(pdf_path)
            if paper.paper_content:
                logger.info(f"    成功提取 {len(paper.paper_content)} 字符")
                return
        
        logger.debug(f"  未找到论文 {paper.paper_id} 的 PDF 文件")
    
    def save_to_json(self, file_path: Union[str, Path]):
        """保存数据到JSON文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [paper.to_dict() for paper in self.papers]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到: {file_path}")
    
    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        if not self.papers:
            return {}
        
        total_papers = len(self.papers)
        total_reviews = sum(len(p.reviews) for p in self.papers)
        reviews_per_paper = [len(p.reviews) for p in self.papers]
        
        all_scores = [
            r.actual_score 
            for p in self.papers 
            for r in p.reviews
        ]
        
        stats = {
            "total_papers": total_papers,
            "total_reviews": total_reviews,
            "avg_reviews_per_paper": sum(reviews_per_paper) / len(reviews_per_paper),
            "min_reviews_per_paper": min(reviews_per_paper),
            "max_reviews_per_paper": max(reviews_per_paper),
            "score_range": {
                "min": min(all_scores) if all_scores else 0,
                "max": max(all_scores) if all_scores else 0,
                "mean": sum(all_scores) / len(all_scores) if all_scores else 0,
            }
        }
        
        return stats
    
    def display_statistics(self):
        """显示数据集统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 50)
        print("数据集统计信息")
        print("=" * 50)
        print(f"论文总数: {stats.get('total_papers', 0)}")
        print(f"审稿记录总数: {stats.get('total_reviews', 0)}")
        print(f"平均每篇论文的审稿数: {stats.get('avg_reviews_per_paper', 0):.2f}")
        print(f"最少审稿数: {stats.get('min_reviews_per_paper', 0)}")
        print(f"最多审稿数: {stats.get('max_reviews_per_paper', 0)}")
        
        score_range = stats.get('score_range', {})
        print(f"\n分数范围:")
        print(f"  最低分: {score_range.get('min', 0):.2f}")
        print(f"  最高分: {score_range.get('max', 0):.2f}")
        print(f"  平均分: {score_range.get('mean', 0):.2f}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    # 测试数据加载器
    loader = DataLoader()
    
    # 创建示例数据
    sample_data = [
        {
            "paper_id": "paper_001",
            "title": "Attention Is All You Need",
            "abstract": "We propose a new attention mechanism...",
            "reviews": [
                {
                    "reviewer_id": "reviewer_1",
                    "review_text": "This paper presents an innovative approach...",
                    "actual_score": 8
                },
                {
                    "reviewer_id": "reviewer_2",
                    "review_text": "The paper has some merits but...",
                    "actual_score": 6
                }
            ]
        }
    ]
    
    # 保存示例数据
    sample_file = Path("sample_data.json")
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # 测试加载
    papers = loader.load_from_json(sample_file)
    loader.display_statistics()
    
    # 清理
    sample_file.unlink()
    
    print("✓ 数据加载器测试完成！")


