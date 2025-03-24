"""
پوشه grammar شامل زیرسیستم‌های مربوط به پردازش گرامر زبان فارسی می‌باشد.
این زیرسیستم وظایف زیر را انجام می‌دهد:
  - اصلاح خودکار خطاهای گرامری (CorrectionEngine)
  - تحلیل جامع گرامری متون فارسی (GrammarProcessor)
  - مدیریت قواعد گرامری (RuleManager)
  - تحلیل و اعتبارسنجی برچسب‌های دستوری (POSAnalyzer)
  - جمع‌آوری و گزارش متریک‌های عملکردی گرامری (GrammarMetrics)

این فایل __init__.py تمامی کلاس‌های اصلی موجود در این پوشه را صادر می‌کند تا سایر بخش‌های پروژه به‌راحتی بتوانند از آن‌ها استفاده کنند.
"""

from .correction_engine import CorrectionEngine
from .grammar_processor import GrammarProcessor
from .rule_manager import RuleManager
from .pos_analyzer import POSAnalyzer
from .metrics import GrammarMetrics

__all__ = [
    "CorrectionEngine",
    "GrammarProcessor",
    "RuleManager",
    "POSAnalyzer",
    "GrammarMetrics"
]
