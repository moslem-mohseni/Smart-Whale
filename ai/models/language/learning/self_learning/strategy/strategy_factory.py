"""
StrategyFactory Module
-------------------------
این فایل مسئول انتخاب استراتژی آموزشی مناسب بر اساس فاز مدل (BEGINNER, INTERMEDIATE, ADVANCED) و شرایط اضافی است.
استراتژی‌های مختلف (BeginnerStrategy، IntermediateStrategy، AdvancedStrategy) از قبل پیاده‌سازی شده‌اند و این کارخانه
با استفاده از ورودی "phase" و تنظیمات پیکربندی، نمونه‌ی مناسب را ایجاد می‌کند.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌ها و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from typing import Dict, Any, Optional

from .beginner_strategy import BeginnerStrategy
from .intermediate_strategy import IntermediateStrategy
from .advanced_strategy import AdvancedStrategy


class StrategyFactory:
    """
    StrategyFactory مسئول انتخاب و تولید استراتژی مناسب برای آموزش مدل بر اساس فاز مشخص‌شده است.

    ورودی:
      - phase: فاز مدل به صورت رشته ("BEGINNER"، "INTERMEDIATE"، "ADVANCED")
      - config: دیکشنری تنظیمات اختصاصی برای هر استراتژی (می‌تواند شامل تنظیمات جداگانه برای هر فاز باشد)
      - additional_context (اختیاری): زمینه‌های اضافی برای انتخاب دقیق‌تر استراتژی.

    خروجی:
      - یک نمونه از استراتژی مناسب برای استفاده در فرآیند آموزش.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger("StrategyFactory")
        self.config = config or {}
        self.logger.info(f"[StrategyFactory] Initialized with config: {self.config}")

    def get_strategy(self, phase: str, additional_context: Optional[Dict[str, Any]] = None) -> Any:
        """
        انتخاب و تولید استراتژی مناسب بر اساس فاز مشخص‌شده.

        Args:
            phase (str): فاز مدل ("BEGINNER"، "INTERMEDIATE"، "ADVANCED")
            additional_context (Optional[Dict[str, Any]]): زمینه‌های اضافی جهت تصمیم‌گیری (اختیاری).

        Returns:
            An instance of the selected strategy.
        """
        phase_upper = phase.upper()
        if phase_upper == "BEGINNER":
            strategy_instance = BeginnerStrategy(config=self.config.get("beginner"))
        elif phase_upper == "INTERMEDIATE":
            strategy_instance = IntermediateStrategy(config=self.config.get("intermediate"))
        elif phase_upper == "ADVANCED":
            strategy_instance = AdvancedStrategy(config=self.config.get("advanced"))
        else:
            self.logger.warning(f"[StrategyFactory] Unknown phase '{phase}'. Defaulting to BeginnerStrategy.")
            strategy_instance = BeginnerStrategy(config=self.config.get("beginner"))

        self.logger.info(
            f"[StrategyFactory] Selected strategy for phase {phase_upper}: {strategy_instance.__class__.__name__}")
        return strategy_instance
