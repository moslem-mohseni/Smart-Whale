import time
from core.cache.manager import CacheManager
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resource_management.optimizer.load_balancer import LoadBalancer
from core.resource_management.monitor.threshold_manager import ThresholdManager
from core.monitoring.metrics.collector import MetricsCollector
from core.monitoring.metrics.exporter import MetricsExporter


class LoadPredictor:
    """
    پیش‌بینی بار پردازشی سیستم بر اساس روند مصرف منابع.
    """

    def __init__(self):
        self.cache = CacheManager()
        self.resource_monitor = ResourceMonitor()
        self.load_balancer = LoadBalancer()
        self.threshold_manager = ThresholdManager()
        self.metrics_collector = MetricsCollector()
        self.metrics_exporter = MetricsExporter()

    async def predict_load(self, time_window: int = 60) -> dict:
        """
        پیش‌بینی بار پردازشی سیستم در آینده نزدیک.

        :param time_window: بازه زمانی موردنظر برای پیش‌بینی (به ثانیه)
        :return: دیکشنری شامل میزان بار پردازشی پیش‌بینی‌شده
        """
        cache_key = f"system_load_prediction:{time_window}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        system_load = await self.resource_monitor.get_system_load()
        past_usage = await self.resource_monitor.get_historical_usage(time_window)
        load_trend = self._analyze_trend(past_usage, system_load)

        threshold_data = await self.threshold_manager.get_thresholds("system")

        predicted_load = self._calculate_final_load(load_trend, threshold_data)

        self.cache.set(cache_key, predicted_load, ttl=600)
        self.metrics_collector.record("system_load_prediction", predicted_load)
        self.metrics_exporter.export(predicted_load)

        return predicted_load

    def _analyze_trend(self, past_usage: list, current_load: dict) -> dict:
        """
        تحلیل روند بار پردازشی سیستم.

        :param past_usage: داده‌های گذشته از مصرف منابع
        :param current_load: میزان مصرف کنونی سیستم
        :return: دیکشنری شامل روند بار پردازشی
        """
        avg_cpu_usage = sum(u["cpu"] for u in past_usage) / len(past_usage)
        avg_memory_usage = sum(u["memory"] for u in past_usage) / len(past_usage)

        cpu_trend = (current_load["cpu"] - avg_cpu_usage) / avg_cpu_usage
        memory_trend = (current_load["memory"] - avg_memory_usage) / avg_memory_usage

        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend
        }

    def _calculate_final_load(self, load_trend: dict, threshold_data: dict) -> dict:
        """
        نهایی‌سازی مقدار بار پردازشی پیش‌بینی شده.

        :param load_trend: تحلیل روند مصرف منابع
        :param threshold_data: اطلاعات آستانه‌های مجاز سیستم
        :return: مقدار نهایی بار پردازشی پیش‌بینی شده
        """
        predicted_cpu = min(load_trend["cpu_trend"] * 100, threshold_data["max_cpu"])
        predicted_memory = min(load_trend["memory_trend"] * 100, threshold_data["max_memory"])

        return {
            "predicted_cpu_load": predicted_cpu,
            "predicted_memory_load": predicted_memory,
            "timestamp": time.time()
        }
