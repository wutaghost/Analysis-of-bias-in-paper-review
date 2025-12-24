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
    MAX_TOKENS: int = 8000  # 增大以支持大量优缺点的量化输出
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))  # 重试次数
    RETRY_DELAY: int = 2  # 重试延迟（秒）
    REQUEST_DELAY: float = float(os.getenv("REQUEST_DELAY", "1.0"))  # 请求间隔（秒）
    
    # ========== 批次处理配置 ==========
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))  # 每批处理的论文数量
    BATCH_DELAY: float = float(os.getenv("BATCH_DELAY", "5.0"))  # 批次/步骤之间的延迟（秒）
    CHECKPOINT_INTERVAL: int = int(os.getenv("CHECKPOINT_INTERVAL", "5"))  # 检查点保存间隔
    
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
    DETAILS_DIR: Path = Path("./results/paper_details")
    # 中间文件输出目录
    EXTRACTION_DIR: Path = Path("./results/extraction")  # 步骤1: 原始提取结果
    ANONYMIZED_DIR: Path = Path("./results/anonymized")  # 步骤2: 去审稿人后的文件
    QUANTIFIED_DIR: Path = Path("./results/quantified")  # 步骤3: 量化结果
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
        cls.DETAILS_DIR.mkdir(parents=True, exist_ok=True)
        cls.EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
        cls.ANONYMIZED_DIR.mkdir(parents=True, exist_ok=True)
        cls.QUANTIFIED_DIR.mkdir(parents=True, exist_ok=True)
        
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
    
    # ========== 优缺点提取Prompt（一次性处理一篇论文的所有审稿意见）==========
    EXTRACT_PROS_CONS_BATCH = """你是一位资深的学术论文审稿专家。请仔细阅读以下论文的所有审稿意见，为每位审稿人分别提取优点（Pros）和缺点（Cons）。

论文标题: {title}
论文摘要: {abstract}

以下是 {num_reviewers} 位审稿人的审稿意见:

{all_reviews_text}

要求:
1. 为每位审稿人分别提取优缺点，保持审稿人ID的对应关系
2. 将每个优点和缺点提取出来，简要总结（1-2句话）
3. 为每个优缺点分配一个合适的类别，类别必须从以下列表中选择：
   {categories}
4. 优点和缺点应该具体、可量化
5. 按JSON格式返回结果

返回格式示例（假设有2位审稿人）:
{{
  "reviewers": [
    {{
      "reviewer_id": "Reviewer_A",
      "pros": [
        {{
          "description": "提出了创新的注意力机制",
          "category": "创新性 (Novelty/Originality)"
        }}
      ],
      "cons": [
        {{
          "description": "缺少与最新SOTA方法的对比",
          "category": "实验充分性 (Experimental Rigor)"
        }}
      ]
    }},
    {{
      "reviewer_id": "Reviewer_B",
      "pros": [
        {{
          "description": "实验设计严谨",
          "category": "实验充分性 (Experimental Rigor)"
        }}
      ],
      "cons": [
        {{
          "description": "论文写作存在语法错误",
          "category": "写作质量 (Writing Quality)"
        }}
      ]
    }}
  ]
}}

请严格按照上述JSON格式返回，确保每位审稿人的 reviewer_id 与输入中的ID完全一致。
"""

    # ========== 权重量化Prompt（基于去审稿人信息后的优缺点列表）==========
    QUANTIFY_WEIGHTS = """你是一位客观公正的学术评审专家。现在需要你结合论文全文内容，为审稿中提取出的优缺点赋予量化权重。

论文标题: {title}
论文摘要: {abstract}

论文全文内容 (参考):
{paper_content}

已识别的优点（共 {pros_count} 条）:
{pros_text}

已识别的缺点（共 {cons_count} 条）:
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

**重要要求**:
- 你必须为上述列出的【每一条】优点和缺点分配权重，不能遗漏任何一条
- pros_weights 数组必须恰好包含 {pros_count} 个元素（与输入优点数量一致）
- cons_weights 数组必须恰好包含 {cons_count} 个元素（与输入缺点数量一致）
- 即使某些条目看起来相似或重复，也必须单独为每条分配权重
- 按照输入的顺序依次输出每条的权重

请为每个优缺点分配合理的权重，返回JSON格式:

{{
  "pros_weights": [
    {{
      "description": "优点1描述（与输入第1条对应）",
      "category": "类别",
      "weight": 1.5,
      "reasoning": "权重分配理由"
    }},
    {{
      "description": "优点2描述（与输入第2条对应）",
      "category": "类别",
      "weight": 1.0,
      "reasoning": "权重分配理由"
    }}
  ],
  "cons_weights": [
    {{
      "description": "缺点1描述（与输入第1条对应）",
      "category": "类别",
      "weight": -1.0,
      "reasoning": "权重分配理由"
    }},
    {{
      "description": "缺点2描述（与输入第2条对应）",
      "category": "类别",
      "weight": -0.5,
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

请严格按照上述JSON格式返回，确保：
1. pros_weights 数组恰好有 {pros_count} 个元素
2. cons_weights 数组恰好有 {cons_count} 个元素
3. 所有数值都是浮点数
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


