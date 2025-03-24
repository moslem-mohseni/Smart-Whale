import time
from core.cache.manager import CacheManager
from core.monitoring.metrics.collector import MetricsCollector
from core.monitoring.metrics.exporter import MetricsExporter
from core.utils.math_utils import optimize_parameters


class PredictionTuner:
    """
    تنظیم و بهینه‌سازی مدل‌های پیش‌بینی برای افزایش دقت و عملکرد.
    """

    def __init__(self):
        self.cache = CacheManager()
        self.metrics_collector = MetricsCollector()
        self.metrics_exporter = MetricsExporter()

    async def tune_predictions(self, model_id: str, prediction_data: dict) -> dict:
        """
        بهینه‌سازی پیش‌بینی‌های مدل مشخص شده.

        :param model_id: شناسه مدل مورد بررسی
        :param prediction_data: داده‌های مربوط به پیش‌بینی‌های قبلی
        :return: دیکشنری شامل تنظیمات بهینه شده برای مدل
        """
        cache_key = f"prediction_tuning:{model_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        optimized_parameters = self._adjust_prediction_parameters(prediction_data)

        self.cache.set(cache_key, optimized_parameters, ttl=3600)
        self.metrics_collector.record("prediction_tuning", optimized_parameters)
        self.metrics_exporter.export(optimized_parameters)

        return optimized_parameters

    def _adjust_prediction_parameters(self, prediction_data: dict) -> dict:
        """
        بهینه‌سازی پارامترهای مدل بر اساس دقت و عملکرد گذشته.

        :param prediction_data: داده‌های پیش‌بینی شده قبلی
        :return: پارامترهای بهینه شده برای مدل
        """
        current_accuracy = prediction_data.get("accuracy", 1.0)
        optimized_settings = optimize_parameters(current_accuracy)

        return {
            "optimized_learning_rate": optimized_settings["learning_rate"],
            "optimized_threshold": optimized_settings["threshold"],
            "last_updated": time.time()
        }
