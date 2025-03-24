import numpy as np
from collections import Counter
from core.monitoring.metrics.collector import MetricsCollector
from core.cache.manager.cache_manager import CacheManager


class PatternDetector:
    """
    ماژولی برای تشخیص الگوهای تکراری در داده‌های پردازشی.
    این ماژول با استفاده از تحلیل متریک‌ها و داده‌های ورودی، روندهای مشخص را شناسایی می‌کند.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.cache_manager = CacheManager()
        self.pattern_cache_key = "pattern_detector_cache"

    async def analyze_patterns(self, data_stream: list) -> dict:
        """
        تجزیه و تحلیل الگوهای داده‌های پردازشی.

        :param data_stream: لیستی از داده‌های ورودی که پردازش شده‌اند.
        :return: دیکشنری شامل الگوهای شناسایی‌شده
        """
        if not data_stream:
            return {}

        # بررسی کش برای داده‌های مشابه قبلی
        cached_patterns = await self.cache_manager.get(self.pattern_cache_key)
        if cached_patterns:
            return cached_patterns

        # تبدیل داده‌ها به فرمت تحلیلی
        frequency_distribution = self._calculate_frequencies(data_stream)
        trending_patterns = self._detect_trends(frequency_distribution)

        # ذخیره در کش برای استفاده‌های بعدی
        await self.cache_manager.set(self.pattern_cache_key, trending_patterns, ttl=3600)

        return trending_patterns

    def _calculate_frequencies(self, data_stream: list) -> dict:
        """
        محاسبه توزیع فراوانی الگوها در داده‌های پردازشی.

        :param data_stream: لیستی از داده‌ها
        :return: دیکشنری شامل فرکانس هر الگو
        """
        return dict(Counter(data_stream))

    def _detect_trends(self, frequency_distribution: dict) -> dict:
        """
        تحلیل توزیع فرکانسی داده‌ها و شناسایی الگوهای پرتکرار.

        :param frequency_distribution: دیکشنری از توزیع فراوانی
        :return: دیکشنری شامل الگوهای شناسایی‌شده
        """
        threshold = np.percentile(list(frequency_distribution.values()), 90)
        trending_patterns = {k: v for k, v in frequency_distribution.items() if v >= threshold}
        return trending_patterns
