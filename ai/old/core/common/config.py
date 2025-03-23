# ai/core/common/config.py
"""
تنظیمات مربوط به GPT و سایر پیکربندی‌های عمومی
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """تنظیمات اتصال به ChatGPT"""
    api_key: str
    api_endpoint: str = "https://api.openai.com/v1/chat/completions"
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 2  # ثانیه

    # تنظیمات کش
    cache_enabled: bool = True
    cache_ttl: int = 3600  # یک ساعت

    # تنظیمات یادگیری
    learning_threshold: float = 0.8
    min_samples_for_training: int = 1000

    # تنظیمات مانیتورینگ
    collect_metrics: bool = True
    log_queries: bool = True

    def validate(self) -> bool:
        """اعتبارسنجی تنظیمات"""
        if not self.api_key:
            raise ValueError("API key is required")
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("Max tokens must be positive")
        return True