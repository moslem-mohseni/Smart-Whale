import time
from core.cache.manager import CacheManager
from core.monitoring.metrics.collector import MetricsCollector
from core.monitoring.metrics.exporter import MetricsExporter
from core.utils.math_utils import calculate_error_rate


class AccuracyMonitor:
    """
    پایش دقت پیش‌بینی‌ها و تحلیل عملکرد مدل‌های پیش‌بینی.
    """

    def __init__(self):
        self.cache = CacheManager()
        self.metrics_collector = MetricsCollector()
        self.metrics_exporter = MetricsExporter()

    async def monitor_accuracy(self, model_id: str, actual_data: dict, predicted_data: dict) -> dict:
        """
        ارزیابی دقت پیش‌بینی‌های مدل مشخص شده.

        :param model_id: شناسه مدل مورد بررسی
        :param actual_data: داده‌های واقعی برای مقایسه
        :param predicted_data: داده‌های پیش‌بینی شده توسط مدل
        :return: دیکشنری شامل میزان دقت و تغییرات آن
        """
        cache_key = f"accuracy_monitor:{model_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        error_rate = self._calculate_accuracy(actual_data, predicted_data)

        accuracy_report = {
            "model_id": model_id,
            "error_rate": error_rate,
            "accuracy_score": 1.0 - error_rate,
            "timestamp": time.time()
        }

        self.cache.set(cache_key, accuracy_report, ttl=3600)
        self.metrics_collector.record("accuracy_monitor", accuracy_report)
        self.metrics_exporter.export(accuracy_report)

        return accuracy_report

    def _calculate_accuracy(self, actual_data: dict, predicted_data: dict) -> float:
        """
        محاسبه میزان دقت پیش‌بینی با مقایسه داده‌های واقعی و پیش‌بینی‌شده.

        :param actual_data: داده‌های واقعی
        :param predicted_data: داده‌های پیش‌بینی‌شده
        :return: میزان خطا به‌صورت عددی بین 0 تا 1
        """
        return calculate_error_rate(actual_data, predicted_data)
