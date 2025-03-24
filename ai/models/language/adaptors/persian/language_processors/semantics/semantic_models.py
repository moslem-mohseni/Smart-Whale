"""
ماژول semantic_models.py

در این فایل مدل‌های داده‌ای مرتبط با تحلیل معنایی متون فارسی تعریف شده‌اند.
این مدل‌ها شامل ساختارهایی برای نگهداری نتایج تحلیل معنایی و اطلاعات مربوط به دسته‌بندی موضوعات (Topic Categories) هستند.
از قابلیت‌های dataclass برای تعریف مدل‌های سبک و خوانا استفاده شده است.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import time

@dataclass
class SemanticAnalysisResult:
    """
    مدل نتیجه تحلیل معنایی یک متن
    """
    text: str                             # متن اصلی ورودی
    normalized_text: str                  # متن نرمال‌شده
    intent: str                           # هدف تشخیص داده شده (مانند greeting, question, ...)
    sentiment: str                        # تحلیل احساسات (مثلاً positive, neutral, negative, ...)
    topics: List[str]                     # لیست موضوعات استخراج‌شده
    embedding: List[float]                # بردار معنایی متن
    confidence: float                     # سطح اطمینان تحلیل
    source: str                           # منبع تحلیل (مثلاً smart_model، teacher، rule_based)
    timestamp: float = field(default_factory=time.time)  # زمان ایجاد نتیجه (برحسب timestamp)

@dataclass
class TopicCategory:
    """
    مدل اطلاعات دسته‌بندی موضوع (Topic Category)
    """
    topic_id: str                         # شناسه یکتا برای موضوع
    topic_name: str                       # نام موضوع
    keywords: List[str]                   # لیست کلمات کلیدی مرتبط با موضوع
    parent_topic: Optional[str] = ""      # موضوع والد (در صورت وجود)
    source: str = "default"               # منبع ایجاد یا کشف موضوع (مانند auto_discovery، teacher، manual)
    samples: List[str] = field(default_factory=list)  # نمونه متونی که موضوع بر اساس آنها شناسایی شده
    discovery_time: float = field(default_factory=time.time)  # زمان کشف موضوع (timestamp)
    usage_count: int = 1                  # تعداد دفعات استفاده از موضوع (برای آمارگیری)
