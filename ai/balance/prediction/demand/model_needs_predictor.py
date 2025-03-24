import time
from core.cache.manager import CacheManager
from core.cache.analytics.usage_tracker import UsageTracker
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resource_management.monitor.threshold_manager import ThresholdManager
from core.utils.validation.input_validator import InputValidator
from core.monitoring.metrics.collector import MetricsCollector
from core.monitoring.metrics.exporter import MetricsExporter


class ModelNeedsPredictor:
    """
    پیش‌بینی نیاز داده‌ای مدل‌ها بر اساس الگوی استفاده و مصرف قبلی.
    """

    def __init__(self):
        self.cache = CacheManager()
        self.usage_tracker = UsageTracker()
        self.resource_monitor = ResourceMonitor()
        self.threshold_manager = ThresholdManager()
        self.metrics_collector = MetricsCollector()
        self.metrics_exporter = MetricsExporter()
        self.input_validator = InputValidator()

    async def predict_needs(self, model_id: str, recent_requests: list) -> dict:
        """
        پیش‌بینی نیاز داده‌ای مدل مشخص شده.

        :param model_id: شناسه مدل مورد بررسی
        :param recent_requests: لیست درخواست‌های اخیر مدل
        :return: دیکشنری شامل پیش‌بینی میزان داده موردنیاز
        """
        self.input_validator.validate({"model_id": model_id, "recent_requests": recent_requests})

        cache_key = f"model_needs:{model_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        usage_stats = await self.usage_tracker.get_usage_stats(model_id)
        system_load = await self.resource_monitor.get_system_load()
        threshold_data = await self.threshold_manager.get_thresholds(model_id)

        predicted_needs = self._calculate_needs(model_id, usage_stats, system_load, threshold_data)

        self.cache.set(cache_key, predicted_needs, ttl=600)
        self.metrics_collector.record("model_needs_prediction", predicted_needs)
        self.metrics_exporter.export(predicted_needs)

        return predicted_needs

    def _calculate_needs(self, model_id: str, usage_stats: dict, system_load: dict, threshold_data: dict) -> dict:
        """
        محاسبه مقدار داده‌ای که مدل نیاز خواهد داشت.

        :param model_id: شناسه مدل
        :param usage_stats: آمار مصرف داده در گذشته
        :param system_load: وضعیت بار کنونی سیستم
        :param threshold_data: اطلاعات آستانه‌ها
        :return: مقدار داده پیش‌بینی شده
        """
        base_demand = usage_stats["average_data_consumption"]
        peak_usage = usage_stats["peak_usage"]
        load_factor = system_load["cpu"] * 0.7 + system_load["memory"] * 0.3

        predicted_data_size = base_demand * (1 + load_factor)

        if predicted_data_size > threshold_data["max_data"]:
            predicted_data_size = threshold_data["max_data"]

        return {
            "model_id": model_id,
            "predicted_data_size": predicted_data_size,
            "timestamp": time.time()
        }
