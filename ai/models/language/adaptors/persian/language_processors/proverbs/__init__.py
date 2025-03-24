# persian/language_processors/proverbs/__init__.py
"""
این پوشه شامل زیرسیستم ضرب‌المثل فارسی می‌باشد.
کلاس‌ها و ماژول‌های اصلی شامل:
  - PersianProverbProcessor
  - ProverbDataAccess
  - ProverbVectorManager
  - ProverbMetrics
  - ProverbServices
  - تنظیمات مربوط به ضرب‌المثل (proverb_config)
"""

from .proverb_processor import PersianProverbProcessor
from .proverb_data import ProverbDataAccess
from .proverb_vector import ProverbVectorManager
from .proverb_metrics import ProverbMetrics
from .proverb_services import ProverbServices
from .proverb_config import CONFIG

__all__ = [
    "PersianProverbProcessor",
    "ProverbDataAccess",
    "ProverbVectorManager",
    "ProverbMetrics",
    "ProverbServices",
    "CONFIG"
]
