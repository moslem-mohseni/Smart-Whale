"""
strategy Package
-----------------
این پکیج شامل استراتژی‌های آموزشی مختلف برای مدل‌های خودآموزی است:
  - BeginnerStrategy: استراتژی برای مدل‌های نوپا.
  - IntermediateStrategy: استراتژی برای مدل‌های در حال رشد.
  - AdvancedStrategy: استراتژی برای مدل‌های پخته.
  - StrategyFactory: کارخانه‌ای برای انتخاب استراتژی مناسب بر اساس فاز مدل.

تمامی کلاس‌ها به صورت نهایی و عملیاتی پیاده‌سازی شده‌اند.
"""

from .beginner_strategy import BeginnerStrategy
from .intermediate_strategy import IntermediateStrategy
from .advanced_strategy import AdvancedStrategy
from .strategy_factory import StrategyFactory

__all__ = [
    "BeginnerStrategy",
    "IntermediateStrategy",
    "AdvancedStrategy",
    "StrategyFactory"
]
