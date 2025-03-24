# infrastructure/clickhouse/domain/models.py
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional


@dataclass
class AnalyticsQuery:
    """
    مدل داده‌ای برای تعریف یک پرس‌وجوی تحلیلی

    این مدل حاوی متن کوئری و پارامترهای آن است که به‌عنوان ورودی
    برای سرویس تحلیل داده‌ها استفاده می‌شود.

    Attributes:
        query_text (str): متن کوئری SQL
        params (Dict[str, Any], optional): پارامترهای کوئری برای جلوگیری از SQL Injection
    """
    query_text: str
    params: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class AnalyticsResult:
    """
    مدل داده‌ای برای نگهداری نتیجه یک پرس‌وجوی تحلیلی

    این مدل حاوی کوئری اصلی، نتیجه داده‌ها و خطای احتمالی است.

    Attributes:
        query (AnalyticsQuery): کوئری اصلی
        data (List[Any]): نتیجه کوئری
        error (str, optional): خطای احتمالی
    """
    query: AnalyticsQuery
    data: List[Any]
    error: Optional[str] = None
