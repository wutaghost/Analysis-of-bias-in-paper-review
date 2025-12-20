"""
配置管理模块
管理系统的所有配置参数，包括API密钥、模型参数、缓存设置等
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """系统配置类"""
    
    # ========== API配置 ==========
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-ktR9Qut7TzKPFDUf961QsNARbRyNxusp8Wu1H8EyJ0VjGTsG")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL", "http://35.164.11.19:3887/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    # ========== LLM参数配置 ==========
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = 4000
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: int = 2  # 重试延迟（秒）
    
    # ========== 缓存配置 ==========
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./cache"))
    
    # ========== 评分系统配置 ==========
    BASE_SCORE: float = 5.0  # 基准分数
    MIN_SCORE: float = 1.0   # 最低分数
    MAX_SCORE: float = 10.0  # 最高分数
    
    # ========== 特征提取配置 ==========
    # 优缺点分类标签
    CATEGORIES = [
        "创新性 (Novelty/Originality)",
        "技术正确性 (Technical Correctness)",
        "实验充分性 (Experimental Rigor)",
        "写作质量 (Writing Quality)",
        "相关性 (Relevance)",
        "可重复性 (Reproducibility)",
        "理论贡献 (Theoretical Contribution)",
        "实践价值 (Practical Impact)",
    ]
    
    # ========== 输出配置 ==========
    OUTPUT_DIR: Path = Path("./results")
    FIGURE_DPI: int = 300
    FIGURE_FORMAT: str = "png"
    
    @classmethod
    def validate(cls) -> bool:
        """验证配置是否完整"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "未设置 OPENAI_API_KEY！\n"
                "请设置环境变量或在 .env 文件中配置。"
            )
        
        # 创建必要的目录
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def display(cls):
        """显示当前配置（隐藏敏感信息）"""
        print("=" * 50)
        print("系统配置信息")
        print("=" * 50)
        print(f"模型: {cls.MODEL_NAME}")
        print(f"Temperature: {cls.TEMPERATURE}")
        print(f"缓存启用: {cls.ENABLE_CACHE}")
        print(f"缓存目录: {cls.CACHE_DIR}")
        print(f"输出目录: {cls.OUTPUT_DIR}")
        print(f"评分范围: {cls.MIN_SCORE} - {cls.MAX_SCORE}")
        print(f"基准分数: {cls.BASE_SCORE}")
        print("=" * 50)


# 提示词模板
class PromptTemplates:
    """Prompt模板集合"""
    
    # ========== 优缺点提取Prompt ==========
    EXTRACT_PROS_CONS = """你是一位资深的学术论文审稿专家。请仔细阅读以下审稿意见，提取出其中的所有优点（Pros）和缺点（Cons）。

论文标题: {title}
论文摘要: {abstract}

审稿意见:
{review_text}

要求:
1. 将审稿意见中的每个优点和缺点提取出来，简要总结（1-2句话）
2. 为每个优缺点分配一个合适的类别，类别必须从以下列表中选择：
   {categories}
3. 优点和缺点应该具体、可量化
4. 按JSON格式返回结果

返回格式示例:
{{
  "pros": [
    {{
      "description": "提出了创新的注意力机制，显著提升了模型性能",
      "category": "创新性 (Novelty/Originality)"
    }},
    {{
      "description": "实验设计严谨，在多个数据集上验证了方法的有效性",
      "category": "实验充分性 (Experimental Rigor)"
    }}
  ],
  "cons": [
    {{
      "description": "缺少与最新SOTA方法的对比实验",
      "category": "实验充分性 (Experimental Rigor)"
    }},
    {{
      "description": "论文写作存在语法错误，部分段落逻辑不清晰",
      "category": "写作质量 (Writing Quality)"
    }}
  ]
}}

请严格按照上述JSON格式返回，不要添加任何额外的解释文字。
"""

    # ========== 权重量化Prompt ==========
    QUANTIFY_WEIGHTS = """你是一位客观公正的学术评审专家。现在需要你为论文审稿中的优缺点赋予量化权重。

论文标题: {title}
论文摘要: {abstract}

已识别的优点:
{pros_text}

已识别的缺点:
{cons_text}

评分标准:
- 评分范围: {min_score} 到 {max_score}
- 基准分数: {base_score}（表示一篇中等水平的论文）
- 优点权重（正值）: 表示该优点应该使总分增加多少
- 缺点权重（负值）: 表示该缺点应该使总分减少多少

权重分配原则:
1. 创新性、技术正确性等核心要素应给予较高权重（±0.5 到 ±2.0）
2. 写作质量、格式问题等次要因素给予较低权重（±0.1 到 ±0.5）
3. 致命缺陷（如理论错误）可给予极高负权重（-2.0 或更低）
4. 重大创新可给予极高正权重（+2.0 或更高）
5. 权重总和应合理，使期望分数落在 {min_score} 到 {max_score} 范围内

请为每个优缺点分配合理的权重，返回JSON格式:

{{
  "pros_weights": [
    {{
      "description": "优点描述",
      "category": "类别",
      "weight": 1.5,
      "reasoning": "权重分配理由"
    }}
  ],
  "cons_weights": [
    {{
      "description": "缺点描述",
      "category": "类别",
      "weight": -1.0,
      "reasoning": "权重分配理由"
    }}
  ],
  "expected_score_breakdown": {{
    "base_score": {base_score},
    "total_pros_weight": 0.0,
    "total_cons_weight": 0.0,
    "expected_score": 0.0
  }}
}}

请严格按照上述JSON格式返回，确保所有数值都是浮点数。
"""

    # ========== 优缺点相似度比较Prompt ==========
    COMPARE_REVIEWS = """你是一位学术审稿分析专家。请比较两位审稿人提出的优缺点，判断它们的相似程度。

论文标题: {title}

审稿人1的优缺点:
优点: {pros_1}
缺点: {cons_1}

审稿人2的优缺点:
优点: {pros_2}
缺点: {cons_2}

请分析:
1. 两位审稿人发现的优点有多少重叠？
2. 两位审稿人发现的缺点有多少重叠？
3. 整体相似度评分（0-1，1表示完全相同）

返回JSON格式:
{{
  "pros_similarity": 0.75,
  "cons_similarity": 0.80,
  "overall_similarity": 0.77,
  "common_pros": ["共同发现的优点1", "共同发现的优点2"],
  "common_cons": ["共同发现的缺点1", "共同发现的缺点2"],
  "unique_to_reviewer1": ["仅审稿人1提到的点"],
  "unique_to_reviewer2": ["仅审稿人2提到的点"]
}}
"""


if __name__ == "__main__":
    # 测试配置
    try:
        Config.validate()
        Config.display()
        print("\n✓ 配置验证成功！")
    except ValueError as e:
        print(f"\n✗ 配置验证失败: {e}")


