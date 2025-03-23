"""
IntermediateStrategy Module
-----------------------------
این فایل مسئول پیاده‌سازی استراتژی آموزشی برای مدل‌های در حال رشد (Intermediate) در سیستم خودآموزی است.
در این مرحله، مدل‌ها باید به تدریج از دانش پایه به دانش پیشرفته انتقال یابند؛ به همین دلیل نرخ یادگیری متوسط و اندازه دسته متعادل در نظر گرفته می‌شود.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from abc import ABC
from typing import Dict, Any, Optional

from ..base.base_component import BaseComponent


class IntermediateStrategy(BaseComponent, ABC):
    """
    IntermediateStrategy تنظیمات آموزشی مدل‌های در حال رشد را فراهم می‌کند.

    ویژگی‌ها:
      - نرخ یادگیری متوسط برای بهبود تدریجی.
      - اندازه دسته متعادل جهت بهره‌وری مناسب در به‌روز‌رسانی پارامترها.
      - تعیین اولویت "medium" جهت دسته‌بندی مناسب درخواست‌ها در سیستم.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه IntermediateStrategy.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل:
                - "learning_rate": نرخ یادگیری (پیش‌فرض: 0.005)
                - "batch_size": اندازه دسته (پیش‌فرض: 32)
                - "priority": سطح اولویت (پیش‌فرض: "medium")
        """
        super().__init__(component_type="intermediate_strategy", config=config)
        self.logger = logging.getLogger("IntermediateStrategy")
        self.learning_rate = float(self.config.get("learning_rate", 0.005))
        self.batch_size = int(self.config.get("batch_size", 32))
        self.priority = self.config.get("priority", "medium")
        self.logger.info(
            f"[IntermediateStrategy] Initialized with learning_rate={self.learning_rate}, batch_size={self.batch_size}")

    def apply_strategy(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        اعمال استراتژی آموزشی برای مدل‌های در حال رشد.

        Args:
            model_state (Dict[str, Any]): وضعیت فعلی مدل (مثلاً متریک‌های عملکرد یا تغییرات اخیر).

        Returns:
            Dict[str, Any]: تنظیمات پیشنهادی آموزشی برای مدل، شامل نرخ یادگیری، اندازه دسته، و توضیحات.
        """
        strategy_settings = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "strategy": "intermediate",
            "priority": self.priority,
            "notes": "Apply moderate learning rate and balanced batch size for steady growth and gradual transition from basic to advanced knowledge."
        }
        self.logger.info(f"[IntermediateStrategy] Applied strategy: {strategy_settings}")
        return strategy_settings


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    strategy = IntermediateStrategy(config={"learning_rate": 0.006, "batch_size": 30})
    result = strategy.apply_strategy(model_state={})
    print(result)
