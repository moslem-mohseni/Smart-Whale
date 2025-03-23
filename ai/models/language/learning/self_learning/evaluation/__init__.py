"""
evaluation Package
--------------------
این پکیج شامل ابزارهای ارزیابی عملکرد مدل در سیستم خودآموزی است.
موارد شامل:
  - PerformanceMetrics: جمع‌آوری و گزارش متریک‌های عملکرد.
  - KnowledgeCoverage: ارزیابی پوشش دانشی مدل.
  - LearningEfficiency: ارزیابی کارایی فرآیند یادگیری.
  - ImprovementTracker: پیگیری و ثبت پیشرفت مدل در طول زمان.

تمامی کلاس‌ها به صورت نهایی و عملیاتی پیاده‌سازی شده‌اند.
"""

from .performance_metrics import PerformanceMetrics
from .knowledge_coverage import KnowledgeCoverage
from .learning_efficiency import LearningEfficiency
from .improvement_tracker import ImprovementTracker

__all__ = [
    "PerformanceMetrics",
    "KnowledgeCoverage",
    "LearningEfficiency",
    "ImprovementTracker"
]
