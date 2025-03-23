"""
BeginnerStrategy Module
-------------------------
این فایل مسئول پیاده‌سازی استراتژی آموزشی برای مدل‌های نوپا (BEGINNER) است.
در این مرحله، مدل‌ها باید سریعاً دانش پایه را کسب کنند. این استراتژی شامل تنظیماتی مانند نرخ یادگیری بالا،
اندازه دسته کوچک و تأکید بر داده‌های پرتکرار و موضوعات پایه می‌باشد.

این نسخه نهایی و عملیاتی با بهترین مکانیسم‌ها و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from abc import ABC
from typing import Dict, Any, Optional

from ..base.base_component import BaseComponent


class BeginnerStrategy(BaseComponent, ABC):
    """
    BeginnerStrategy برای مدل‌های نوپا تنظیمات آموزشی را فراهم می‌کند تا مدل به سرعت دانش پایه را کسب کند.

    ویژگی‌های کلیدی:
      - نرخ یادگیری بالا برای یادگیری سریع.
      - اندازه دسته کوچک برای کاهش پیچیدگی به‌روز‌رسانی.
      - اولویت بالا برای داده‌های اولیه و پرتکرار.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        راه‌اندازی اولیه BeginnerStrategy.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل:
                - "learning_rate": نرخ یادگیری (پیش‌فرض: 0.01)
                - "batch_size": اندازه دسته (پیش‌فرض: 16)
                - "priority": سطح اولویت (پیش‌فرض: "high")
        """
        super().__init__(component_type="beginner_strategy", config=config)
        self.logger = logging.getLogger("BeginnerStrategy")
        self.learning_rate = float(self.config.get("learning_rate", 0.01))
        self.batch_size = int(self.config.get("batch_size", 16))
        self.priority = self.config.get("priority", "high")
        self.logger.info(
            f"[BeginnerStrategy] Initialized with learning_rate={self.learning_rate}, batch_size={self.batch_size}")

    def apply_strategy(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        اعمال استراتژی آموزشی برای مدل‌های نوپا.

        Args:
            model_state (Dict[str, Any]): وضعیت فعلی مدل (مثلاً متریک‌های عملکرد، نرخ بهبود و ...)

        Returns:
            Dict[str, Any]: تنظیمات پیشنهادی آموزشی برای مدل.
        """
        # در این استراتژی، نرخ یادگیری بالا و اندازه دسته کوچک حفظ می‌شود.
        strategy_settings = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "strategy": "beginner",
            "priority": self.priority,
            "notes": "Apply high learning rate and small batch size for rapid initial learning and focus on core topics."
        }
        self.logger.info(f"[BeginnerStrategy] Applied strategy: {strategy_settings}")
        return strategy_settings


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    strategy = BeginnerStrategy(config={"learning_rate": 0.02, "batch_size": 12})
    result = strategy.apply_strategy(model_state={})
    print(result)
