"""
工具函数模块
提供缓存、日志、JSON解析等通用功能
"""

import json
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps
import time
import logging
from datetime import datetime

from config import Config


# ========== 日志配置 ==========
def setup_logger(name: str = "ReviewBias") -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 文件处理器
    log_dir = Config.OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


logger = setup_logger()


# ========== 缓存管理 ==========
class CacheManager:
    """缓存管理器，用于缓存LLM API调用结果"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = Config.ENABLE_CACHE
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        # 将参数序列化为字符串（只保留可序列化的参数）
        serializable_args = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                serializable_args.append(arg)
            else:
                # 对于不可序列化的对象，使用其字符串表示
                serializable_args.append(str(arg)[:100])
        
        serializable_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                serializable_kwargs[k] = v
            else:
                serializable_kwargs[k] = str(v)[:100]
        
        key_str = json.dumps(
            {"args": serializable_args, "kwargs": serializable_kwargs}, 
            sort_keys=True
        )
        # 使用MD5生成哈希
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if not self.enabled:
            return None
        
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"缓存命中: {key}")
                return data
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
                return None
        return None
    
    def set(self, key: str, value: Any) -> None:
        """保存数据到缓存"""
        if not self.enabled:
            return
        
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"已缓存: {key}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def clear(self) -> None:
        """清空所有缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("已清空所有缓存")


# 全局缓存管理器实例
cache_manager = CacheManager()


def cached(func: Callable) -> Callable:
    """
    缓存装饰器
    用于自动缓存函数调用结果
    注意：自动跳过 self 参数（用于类方法）
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 跳过 self 参数（如果是类方法）
        cache_args = args
        if args and hasattr(args[0], '__class__'):
            # 第一个参数是对象实例，跳过它
            cache_args = args[1:]
        
        # 生成缓存键
        cache_key = cache_manager._get_cache_key(
            func.__name__, *cache_args, **kwargs
        )
        
        # 尝试从缓存获取
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 保存到缓存
        cache_manager.set(cache_key, result)
        
        return result
    
    return wrapper


# ========== 重试机制 ==========
def retry_on_failure(max_retries: int = None, delay: int = None):
    """
    重试装饰器
    用于API调用失败时自动重试，使用指数退避策略
    对于401错误（速率限制），等待更长时间
    """
    max_retries = max_retries or Config.MAX_RETRIES
    delay = delay or Config.RETRY_DELAY
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e)
                    
                    # 检测是否是速率限制错误 (401, 429, rate limit)
                    is_rate_limit = any(x in error_str.lower() for x in ['401', '429', 'rate', 'limit', '无效的令牌'])
                    
                    if is_rate_limit:
                        # 速率限制：使用更长的退避时间
                        wait_time = delay * (2 ** attempt) + 5  # 指数退避 + 额外5秒
                        logger.warning(
                            f"检测到速率限制，第 {attempt + 1}/{max_retries} 次尝试失败，"
                            f"等待 {wait_time} 秒后重试: {e}"
                        )
                    else:
                        # 普通错误：标准递增延迟
                        wait_time = delay * (attempt + 1)
                    logger.warning(
                        f"第 {attempt + 1}/{max_retries} 次尝试失败: {e}"
                    )
                    
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
            
            logger.error(f"所有重试均失败: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator


# ========== JSON解析 ==========
def safe_json_parse(text: str, default: Any = None) -> Any:
    """
    安全解析JSON字符串
    处理LLM输出中可能包含的额外文本
    """
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取JSON代码块
    import re
    
    # 匹配 ```json ... ``` 或 ``` ... ```
    json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_blocks:
        try:
            return json.loads(json_blocks[0])
        except json.JSONDecodeError:
            pass
    
    # 尝试查找第一个完整的JSON对象
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(text[start_idx:i+1])
                except json.JSONDecodeError:
                    start_idx = -1
    
    logger.warning(f"无法解析JSON，返回默认值: {text[:100]}...")
    return default


# ========== 进度跟踪 ==========
class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """更新进度"""
        self.current += n
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            logger.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({percentage:.1f}%) - ETA: {eta:.0f}秒"
            )
    
    def finish(self):
        """完成"""
        elapsed = time.time() - self.start_time
        logger.info(
            f"{self.description}: 完成! "
            f"总耗时: {elapsed:.1f}秒"
        )


# ========== 统计函数 ==========
def calculate_statistics(values: list) -> dict:
    """计算统计指标"""
    import numpy as np
    
    values = np.array(values)
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


# ========== 文本处理 ==========
def truncate_text(text: str, max_length: int = 100) -> str:
    """截断文本，添加省略号"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    从PDF文件中提取文本
    使用 PyPDF2
    """
    try:
        from PyPDF2 import PdfReader
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        logger.warning("未安装 PyPDF2，无法从 PDF 提取文本。请运行: pip install PyPDF2")
        return ""
    except Exception as e:
        logger.error(f"从 PDF {pdf_path} 提取文本失败: {e}")
        return ""


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    估算token数量
    注意：这是简单估算，实际token数可能有差异
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # 简单估算：1 token ≈ 4 字符（英文）或 1.5 字符（中文）
        return len(text) // 2


if __name__ == "__main__":
    # 测试工具函数
    print("测试日志系统...")
    logger.info("这是一条info日志")
    logger.warning("这是一条warning日志")
    
    print("\n测试缓存系统...")
    cache_manager.set("test_key", {"data": "test"})
    result = cache_manager.get("test_key")
    print(f"缓存测试结果: {result}")
    
    print("\n测试JSON解析...")
    test_json = '```json\n{"key": "value"}\n```'
    parsed = safe_json_parse(test_json)
    print(f"JSON解析结果: {parsed}")
    
    print("\n✓ 工具函数测试完成！")


