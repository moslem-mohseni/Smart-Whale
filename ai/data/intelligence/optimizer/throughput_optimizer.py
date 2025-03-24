import numpy as np
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.cache.manager.cache_manager import CacheManager

class ThroughputOptimizer:
    """
    ماژولی برای بهینه‌سازی توان عملیاتی پردازش‌های داده‌ای.
    این ماژول میزان پردازش را بررسی کرده و پیشنهادات بهینه‌سازی ارائه می‌دهد.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
        self.cache_manager = CacheManager()
        self.optimization_thresholds = {
            "min_throughput": 500,  # اگر توان عملیاتی کمتر از 500 پردازش در ثانیه باشد، نیاز به بهینه‌سازی داریم.
            "cpu_usage_limit": 0.80,  # اگر مصرف CPU بیش از 80% شد، پردازش‌های همزمان را کاهش دهد.
            "queue_latency": 200  # اگر تأخیر در صف بیشتر از 200 میلی‌ثانیه باشد، هشدار دهد.
        }

    async def optimize_throughput(self) -> dict:
        """
        بررسی و بهینه‌سازی توان عملیاتی پردازش داده‌ها.

        :return: دیکشنری شامل پیشنهادات بهینه‌سازی.
        """
        # جمع‌آوری متریک‌های پردازشی
        throughput = await self.metrics_collector.get_metric("throughput")
        cpu_usage = await self.resource_monitor.get_cpu_usage()
        queue_latency = await self.metrics_collector.get_metric("queue_latency")

        # تحلیل و بهینه‌سازی توان عملیاتی
        optimization_report = self._analyze_throughput_performance(throughput, cpu_usage, queue_latency)

        return optimization_report

    def _analyze_throughput_performance(self, throughput: list, cpu_usage: list, queue_latency: list) -> dict:
        """
        تحلیل عملکرد توان عملیاتی پردازش‌ها و ارائه راهکارهای بهینه‌سازی.

        :param throughput: میزان پردازش داده‌ها در واحد زمان.
        :param cpu_usage: میزان مصرف CPU در پردازش‌ها.
        :param queue_latency: تأخیر در صف پردازش داده‌ها.
        :return: دیکشنری شامل پیشنهادات بهینه‌سازی.
        """
        optimization_suggestions = {}

        # بررسی توان عملیاتی پردازش‌ها
        avg_throughput = np.mean(throughput) if throughput else 0
        if avg_throughput < self.optimization_thresholds["min_throughput"]:
            optimization_suggestions["increase_throughput"] = {
                "avg_throughput": avg_throughput,
                "suggestion": "افزایش تعداد پردازش‌های موازی یا بهینه‌سازی الگوریتم پردازشی."
            }

        # بررسی مصرف CPU و تنظیم میزان پردازش
        avg_cpu_usage = np.mean(cpu_usage) if cpu_usage else 0
        if avg_cpu_usage > self.optimization_thresholds["cpu_usage_limit"]:
            optimization_suggestions["cpu_overuse"] = {
                "avg_usage": avg_cpu_usage,
                "suggestion": "کاهش تعداد پردازش‌های همزمان یا افزایش تخصیص منابع پردازشی."
            }

        # بررسی تأخیر در صف پردازش
        avg_queue_latency = np.mean(queue_latency) if queue_latency else 0
        if avg_queue_latency > self.optimization_thresholds["queue_latency"]:
            optimization_suggestions["queue_latency_issue"] = {
                "avg_latency": avg_queue_latency,
                "suggestion": "بهینه‌سازی صف‌های پردازشی برای کاهش زمان انتظار."
            }

        return optimization_suggestions
