import time
from core.cache.manager import CacheManager
from core.monitoring.metrics.collector import MetricsCollector
from core.monitoring.metrics.exporter import MetricsExporter
from core.utils.math_utils import optimize_model_parameters


class ModelOptimizer:
    """
    بهینه‌سازی مدل‌های پیش‌بینی بر اساس دقت و عملکرد آن‌ها.
    """

    def __init__(self):
        self.cache = CacheManager()
        self.metrics_collector = MetricsCollector()
        self.metrics_exporter = MetricsExporter()

    async def optimize_model(self, model_id: str, accuracy_data: dict) -> dict:
        """
        تنظیم و بهینه‌سازی مدل پیش‌بینی برای بهبود عملکرد.

        :param model_id: شناسه مدل مورد بررسی
        :param accuracy_data: داده‌های مربوط به عملکرد مدل
        :return: دیکشنری شامل تنظیمات بهینه شده برای مدل
        """
        cache_key = f"model_optimization:{model_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        optimized_parameters = self._adjust_model_parameters(accuracy_data)

        self.cache.set(cache_key, optimized_parameters, ttl=3600)
        self.metrics_collector.record("model_optimization", optimized_parameters)
        self.metrics_exporter.export(optimized_parameters)

        return optimized_parameters

    def _adjust_model_parameters(self, accuracy_data: dict) -> dict:
        """
        اصلاح پارامترهای مدل بر اساس دقت و عملکرد آن.

        :param accuracy_data: داده‌های دقت مدل
        :return: پارامترهای بهینه شده برای مدل
        """
        optimized_settings = optimize_model_parameters(accuracy_data)

        return {
            "optimized_learning_rate": optimized_settings["learning_rate"],
            "optimized_threshold": optimized_settings["threshold"],
            "adjustment_time": time.time()
        }
