import time
from core.cache.manager import CacheManager
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resource_management.monitor.threshold_manager import ThresholdManager
from core.monitoring.metrics.collector import MetricsCollector
from core.monitoring.metrics.exporter import MetricsExporter
from core.utils.math_utils import calculate_trend


class TrendDetector:
    """
    شناسایی روندهای بلندمدت مصرف منابع توسط مدل‌ها.
    """

    def __init__(self):
        self.cache = CacheManager()
        self.resource_monitor = ResourceMonitor()
        self.threshold_manager = ThresholdManager()
        self.metrics_collector = MetricsCollector()
        self.metrics_exporter = MetricsExporter()

    async def detect_trend(self, model_id: str, time_window: int = 86400) -> dict:
        """
        شناسایی روند مصرف منابع مدل در یک بازه زمانی بلندمدت.

        :param model_id: شناسه مدل مورد بررسی
        :param time_window: بازه زمانی برای تحلیل (به ثانیه، پیش‌فرض 24 ساعت)
        :return: دیکشنری شامل تحلیل روند مصرف مدل
        """
        cache_key = f"trend_analysis:{model_id}:{time_window}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        historical_usage = await self.resource_monitor.get_historical_usage(model_id, time_window)
        threshold_data = await self.threshold_manager.get_thresholds(model_id)

        trend_analysis = self._analyze_trend(historical_usage, threshold_data)

        self.cache.set(cache_key, trend_analysis, ttl=3600)
        self.metrics_collector.record("trend_analysis", trend_analysis)
        self.metrics_exporter.export(trend_analysis)

        return trend_analysis

    def _analyze_trend(self, historical_usage: list, threshold_data: dict) -> dict:
        """
        تحلیل داده‌های گذشته برای شناسایی روندهای مصرف.

        :param historical_usage: اطلاعات مصرف مدل در بازه زمانی مشخص
        :param threshold_data: محدودیت‌های تعیین‌شده برای مدل
        :return: تحلیل نهایی روند مصرف مدل
        """
        cpu_trend = calculate_trend([data["cpu"] for data in historical_usage])
        memory_trend = calculate_trend([data["memory"] for data in historical_usage])
        storage_trend = calculate_trend([data["storage"] for data in historical_usage])

        trend_status = "STABLE"
        if cpu_trend > threshold_data["cpu_trend_limit"]:
            trend_status = "INCREASING"
        elif cpu_trend < -threshold_data["cpu_trend_limit"]:
            trend_status = "DECREASING"

        return {
            "model_id": historical_usage[0]["model_id"] if historical_usage else None,
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "storage_trend": storage_trend,
            "trend_status": trend_status,
            "timestamp": time.time()
        }
