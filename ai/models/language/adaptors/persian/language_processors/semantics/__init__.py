"""
ماژول semantics

این زیرسیستم مسئول تحلیل معنایی متون فارسی است.
کلاس‌ها و توابع اصلی این زیرسیستم شامل موارد زیر می‌باشند:
- SemanticServices: ارائه توابع تحلیل معنایی و استخراج ویژگی‌های معنایی
- SemanticMetrics: جمع‌آوری و گزارش متریک‌های تحلیل معنایی
- SemanticProcessor: هماهنگ‌کننده نهایی تحلیل معنایی که خروجی یکپارچه‌ای ارائه می‌دهد
- SemanticAnalysisResult: مدل داده‌ای برای نگهداری نتایج تحلیل معنایی
"""

from .semantic_config import CONFIG
from .semantic_services import SemanticServices
from .semantic_metrics import SemanticMetrics
from .semantic_processor import SemanticProcessor
from .semantic_models import SemanticAnalysisResult

__all__ = [
    "SemanticServices",
    "SemanticMetrics",
    "SemanticProcessor",
    "SemanticAnalysisResult",
    "CONFIG"
]
