"""
این پوشه زیرسیستم حوزه (Domain) را شامل می‌شود.
ماژول‌های موجود در این زیرسیستم:
  - domain_data: دسترسی به داده‌های حوزه (CRUD روی حوزه‌ها، مفاهیم، روابط و ویژگی‌ها)
  - domain_models: مدل‌های داده‌ای حوزه (تعریف کلاس‌های Domain، Concept، Relation و Attribute)
  - domain_services: توابع منطق تجاری حوزه (افزودن، به‌روزرسانی، ادغام و مدیریت حوزه و مفاهیم)
  - domain_analysis: توابع تحلیل دانش حوزه‌ای (یافتن حوزه مرتبط، کشف حوزه و مفاهیم جدید، تولید سلسله‌مراتب، پاسخ به پرسش‌ها، دریافت بردار معنایی و یافتن مفاهیم مشابه)
  - domain_metrics: جمع‌آوری و گزارش متریک‌های عملکردی حوزه
  - domain_config: تنظیمات اختصاصی حوزه (آستانه‌های شباهت، TTL کش، پارامترهای مدل و غیره)
  - domain_processor: هماهنگ‌کننده نهایی که اطلاعات حوزه، مفاهیم، روابط، پاسخ به پرسش‌های حوزه‌ای و متریک‌ها را در یک خروجی یکپارچه ارائه می‌دهد
"""

from .domain_data import DomainDataAccess
from .domain_models import *
from .domain_services import *
from .domain_analysis import DomainAnalysis
from .domain_metrics import DomainMetrics
from .domain_config import DOMAIN_CONFIG
from .domain_processor import DomainProcessor

__all__ = [
    "DomainDataAccess",
    "Domain",
    "Concept",
    "Relation",
    "Attribute",
    "DomainServices",
    "DomainAnalysis",
    "DomainMetrics",
    "DOMAIN_CONFIG",
    "DomainProcessor",
]
