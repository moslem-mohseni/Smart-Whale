"""
need_detection Package
----------------------
این پکیج شامل کلاس‌های اصلی تشخیص نیازهای یادگیری است:
  - NeedDetectorBase: کلاس پایه برای تشخیص نیازها.
  - PerformanceAnalyzer: تحلیل عملکرد مدل برای شناسایی نیازهای یادگیری.
  - GapAnalyzer: شناسایی شکاف‌های دانشی.
  - TrendDetector: تشخیص روندهای داغ در داده‌ها.
  - QueryAnalyzer: تحلیل درخواست‌های متنی کاربران.
  - FeedbackAnalyzer: تحلیل بازخوردهای کاربران جهت شناسایی نیازهای بهبود.

تمامی کلاس‌ها به صورت نهایی و عملیاتی پیاده‌سازی شده‌اند.
"""

from .need_detector_base import NeedDetectorBase
from .performance_analyzer import PerformanceAnalyzer
from .gap_analyzer import GapAnalyzer
from .trend_detector import TrendDetector
from .query_analyzer import QueryAnalyzer
from .feedback_analyzer import FeedbackAnalyzer

__all__ = [
    "NeedDetectorBase",
    "PerformanceAnalyzer",
    "GapAnalyzer",
    "TrendDetector",
    "QueryAnalyzer",
    "FeedbackAnalyzer"
]
