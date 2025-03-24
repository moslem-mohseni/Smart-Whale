import time
from core.cache.manager import CacheManager
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resource_management.monitor.threshold_manager import ThresholdManager
from core.monitoring.metrics.collector import MetricsCollector
from core.monitoring.metrics.exporter import MetricsExporter


class UsageAnalyzer:
    """
    تحلیل روند مصرف منابع و داده‌ها توسط مدل‌ها.
    """

    def __init__(self):
        self.cache = CacheManager()
        self.resource_monitor = ResourceMonitor()
        self.threshold_manager = ThresholdManager()
        self.metrics_collector = MetricsCollector()
        self.metrics_exporter = MetricsExporter()

    async def analyze_usage(self, model_id: str, time_window: int = 60) -> dict:
        """
        تحلیل مصرف داده و منابع برای یک مدل مشخص در یک بازه زمانی.

        :param model_id: شناسه مدل مورد بررسی
        :param time_window: بازه زمانی برای بررسی (به ثانیه)
        :return: دیکشنری شامل تحلیل مصرف منابع
        """
        cache_key = f"usage_analysis:{model_id}:{time_window}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        system_usage = await self.resource_monitor.get_model_usage(model_id, time_window)
        threshold_data = await self.threshold_manager.get_thresholds(model_id)

        usage_analysis = self._process_usage(system_usage, threshold_data)

        self.cache.set(cache_key, usage_analysis, ttl=600)
        self.metrics_collector.record("usage_analysis", usage_analysis)
        self.metrics_exporter.export(usage_analysis)

        return usage_analysis

    def _process_usage(self, system_usage: dict, threshold_data: dict) -> dict:
        """
        پردازش داده‌های مصرفی برای تحلیل دقیق‌تر.

        :param system_usage: اطلاعات مصرف مدل در بازه زمانی مشخص
        :param threshold_data: محدودیت‌های تعیین‌شده برای مدل
        :return: تحلیل نهایی میزان مصرف مدل
        """
        avg_cpu = system_usage["cpu"]
        avg_memory = system_usage["memory"]
        avg_storage = system_usage["storage"]

        cpu_status = "NORMAL"
        memory_status = "NORMAL"
        storage_status = "NORMAL"

        if avg_cpu > threshold_data["max_cpu"]:
            cpu_status = "OVERLOAD"
        if avg_memory > threshold_data["max_memory"]:
            memory_status = "OVERLOAD"
        if avg_storage > threshold_data["max_storage"]:
            storage_status = "OVERLOAD"

        return {
            "model_id": system_usage["model_id"],
            "cpu_usage": avg_cpu,
            "memory_usage": avg_memory,
            "storage_usage": avg_storage,
            "cpu_status": cpu_status,
            "memory_status": memory_status,
            "storage_status": storage_status,
            "timestamp": time.time()
        }
