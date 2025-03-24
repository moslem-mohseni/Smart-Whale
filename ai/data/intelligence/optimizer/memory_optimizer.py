import numpy as np
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.cache.manager.cache_manager import CacheManager

class MemoryOptimizer:
    """
    ماژولی برای بهینه‌سازی مصرف حافظه (RAM) در پردازش‌های داده‌ای.
    این ماژول میزان مصرف حافظه را بررسی کرده و پیشنهادات بهینه‌سازی ارائه می‌دهد.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
        self.cache_manager = CacheManager()
        self.optimization_thresholds = {
            "max_memory_usage": 0.85,  # اگر مصرف RAM بیشتر از 85% شد، باید بهینه‌سازی کنیم.
            "cache_efficiency": 0.70,  # اگر کش کمتر از 70% کارایی داشته باشد، نیاز به بهینه‌سازی دارد.
            "swap_usage": 0.50  # اگر استفاده از Swap بیش از 50% شد، نشان‌دهنده فشار روی RAM است.
        }

    async def optimize_memory(self) -> dict:
        """
        بررسی و بهینه‌سازی مصرف حافظه (RAM).

        :return: دیکشنری شامل پیشنهادات بهینه‌سازی.
        """
        # جمع‌آوری متریک‌های حافظه
        memory_usage = await self.resource_monitor.get_memory_usage()
        cache_efficiency = await self.metrics_collector.get_metric("cache_efficiency")
        swap_usage = await self.metrics_collector.get_metric("swap_usage")

        # تحلیل و بهینه‌سازی مصرف حافظه
        optimization_report = self._analyze_memory_performance(memory_usage, cache_efficiency, swap_usage)

        return optimization_report

    def _analyze_memory_performance(self, memory_usage: list, cache_efficiency: list, swap_usage: list) -> dict:
        """
        تحلیل مصرف حافظه و ارائه راهکارهای بهینه‌سازی.

        :param memory_usage: میزان مصرف RAM
        :param cache_efficiency: میزان کارایی کش
        :param swap_usage: میزان استفاده از Swap
        :return: دیکشنری شامل پیشنهادات بهینه‌سازی.
        """
        optimization_suggestions = {}

        # بررسی مصرف حافظه
        avg_memory_usage = np.mean(memory_usage) if memory_usage else 0
        if avg_memory_usage > self.optimization_thresholds["max_memory_usage"]:
            optimization_suggestions["memory_overuse"] = {
                "avg_usage": avg_memory_usage,
                "suggestion": "کاهش حجم داده‌های پردازشی یا افزایش استفاده از کش."
            }

        # بررسی کارایی کش
        avg_cache_efficiency = np.mean(cache_efficiency) if cache_efficiency else 0
        if avg_cache_efficiency < self.optimization_thresholds["cache_efficiency"]:
            optimization_suggestions["cache_inefficiency"] = {
                "avg_efficiency": avg_cache_efficiency,
                "suggestion": "بهینه‌سازی سیاست‌های کش و افزایش ذخیره‌سازی موقت داده‌های پرتکرار."
            }

        # بررسی میزان استفاده از Swap
        avg_swap_usage = np.mean(swap_usage) if swap_usage else 0
        if avg_swap_usage > self.optimization_thresholds["swap_usage"]:
            optimization_suggestions["high_swap_usage"] = {
                "avg_swap": avg_swap_usage,
                "suggestion": "افزایش ظرفیت RAM یا کاهش حجم پردازش‌های همزمان."
            }

        return optimization_suggestions
