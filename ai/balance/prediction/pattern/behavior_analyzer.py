import time
from core.cache.manager import CacheManager
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resource_management.monitor.threshold_manager import ThresholdManager
from core.monitoring.metrics.collector import MetricsCollector
from core.monitoring.metrics.exporter import MetricsExporter


class BehaviorAnalyzer:
    """
    تحلیل رفتار مدل‌ها و شناسایی الگوهای غیرعادی در مصرف منابع.
    """

    def __init__(self):
        self.cache = CacheManager()
        self.resource_monitor = ResourceMonitor()
        self.threshold_manager = ThresholdManager()
        self.metrics_collector = MetricsCollector()
        self.metrics_exporter = MetricsExporter()

    async def analyze_behavior(self, model_id: str, time_window: int = 300) -> dict:
        """
        تحلیل رفتار مدل مشخص شده در یک بازه زمانی.

        :param model_id: شناسه مدل مورد بررسی
        :param time_window: بازه زمانی تحلیل (به ثانیه)
        :return: دیکشنری شامل تحلیل رفتار مدل
        """
        cache_key = f"behavior_analysis:{model_id}:{time_window}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        system_usage = await self.resource_monitor.get_model_usage(model_id, time_window)
        threshold_data = await self.threshold_manager.get_thresholds(model_id)

        behavior_analysis = self._detect_behavior_patterns(system_usage, threshold_data)

        self.cache.set(cache_key, behavior_analysis, ttl=600)
        self.metrics_collector.record("behavior_analysis", behavior_analysis)
        self.metrics_exporter.export(behavior_analysis)

        return behavior_analysis

    def _detect_behavior_patterns(self, system_usage: dict, threshold_data: dict) -> dict:
        """
        تحلیل داده‌های مصرفی مدل برای شناسایی رفتارهای غیرعادی.

        :param system_usage: اطلاعات مصرف مدل در بازه زمانی مشخص
        :param threshold_data: محدودیت‌های تعیین‌شده برای مدل
        :return: تحلیل نهایی رفتار مدل
        """
        cpu_variation = system_usage["cpu"] - threshold_data["average_cpu"]
        memory_variation = system_usage["memory"] - threshold_data["average_memory"]
        storage_variation = system_usage["storage"] - threshold_data["average_storage"]

        behavior_status = "NORMAL"
        if abs(cpu_variation) > threshold_data["cpu_deviation_limit"]:
            behavior_status = "ANOMALY"
        if abs(memory_variation) > threshold_data["memory_deviation_limit"]:
            behavior_status = "ANOMALY"
        if abs(storage_variation) > threshold_data["storage_deviation_limit"]:
            behavior_status = "ANOMALY"

        return {
            "model_id": system_usage["model_id"],
            "cpu_variation": cpu_variation,
            "memory_variation": memory_variation,
            "storage_variation": storage_variation,
            "behavior_status": behavior_status,
            "timestamp": time.time()
        }
