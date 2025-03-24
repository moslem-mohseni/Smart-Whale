import numpy as np
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.cache.manager.cache_manager import CacheManager

class StreamOptimizer:
    """
    ماژولی برای بهینه‌سازی پردازش جریانی داده‌ها.
    این ماژول با تنظیم نرخ پردازش و تخصیص منابع، کارایی پردازش جریانی را بهبود می‌بخشد.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
        self.cache_manager = CacheManager()
        self.optimization_thresholds = {
            "max_latency": 100,  # اگر تأخیر پردازش از 100 میلی‌ثانیه بیشتر شد، نرخ پردازش را تنظیم کند.
            "cpu_usage_limit": 0.85,  # اگر مصرف CPU بیش از 85% شد، پردازش را محدود کند.
            "buffer_threshold": 0.75  # اگر بافر پردازش به 75% ظرفیت رسید، هشدار بدهد.
        }

    async def optimize_stream(self, stream_data: list) -> dict:
        """
        بررسی و بهینه‌سازی پردازش جریانی.

        :param stream_data: لیستی از داده‌های جریانی که در حال پردازش هستند.
        :return: دیکشنری شامل پیشنهادات بهینه‌سازی پردازش.
        """
        if not stream_data:
            return {"status": "no_data", "message": "هیچ داده‌ای برای بهینه‌سازی وجود ندارد."}

        # جمع‌آوری متریک‌های پردازشی
        processing_latency = await self.metrics_collector.get_metric("processing_latency")
        cpu_usage = await self.resource_monitor.get_cpu_usage()
        buffer_usage = await self.metrics_collector.get_metric("buffer_usage")

        # تحلیل و بهینه‌سازی نرخ پردازش
        optimization_report = self._analyze_stream_performance(
            processing_latency, cpu_usage, buffer_usage
        )

        return optimization_report

    def _analyze_stream_performance(self, processing_latency: list, cpu_usage: list, buffer_usage: list) -> dict:
        """
        تحلیل عملکرد پردازش جریانی و ارائه راهکارهای بهینه‌سازی.

        :param processing_latency: میزان تأخیر در پردازش داده‌های جریانی.
        :param cpu_usage: میزان مصرف CPU در پردازش جریان داده‌ها.
        :param buffer_usage: میزان پر شدن بافر پردازشی.
        :return: دیکشنری شامل پیشنهادات بهینه‌سازی.
        """
        optimization_suggestions = {}

        # تحلیل تأخیر پردازش داده‌ها
        avg_latency = np.mean(processing_latency) if processing_latency else 0
        if avg_latency > self.optimization_thresholds["max_latency"]:
            optimization_suggestions["reduce_latency"] = {
                "avg_latency": avg_latency,
                "suggestion": "افزایش پردازش موازی یا کاهش حجم داده‌های ورودی."
            }

        # بررسی مصرف CPU و تنظیم میزان پردازش
        avg_cpu_usage = np.mean(cpu_usage) if cpu_usage else 0
        if avg_cpu_usage > self.optimization_thresholds["cpu_usage_limit"]:
            optimization_suggestions["cpu_overuse"] = {
                "avg_usage": avg_cpu_usage,
                "suggestion": "کاهش تعداد پردازش‌های همزمان یا افزایش تخصیص منابع پردازشی."
            }

        # بررسی میزان استفاده از بافر
        avg_buffer_usage = np.mean(buffer_usage) if buffer_usage else 0
        if avg_buffer_usage > self.optimization_thresholds["buffer_threshold"]:
            optimization_suggestions["buffer_overload"] = {
                "avg_usage": avg_buffer_usage,
                "suggestion": "بهینه‌سازی استراتژی مدیریت بافر و جلوگیری از سرریز داده‌ها."
            }

        return optimization_suggestions
