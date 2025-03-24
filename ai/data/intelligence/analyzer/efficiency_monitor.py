import numpy as np
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor

class EfficiencyMonitor:
    """
    ماژولی برای پایش مداوم کارایی پردازش‌های داده‌ای و ارائه پیشنهادات بهینه‌سازی.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
        self.efficiency_thresholds = {
            "cpu_efficiency": 0.70,  # اگر پردازش‌ها زیر 70% کارایی داشته باشند، هشدار بدهد
            "memory_efficiency": 0.75,  # اگر مصرف حافظه بهینه نباشد، هشدار بدهد
            "throughput_efficiency": 0.80  # توان عملیاتی پردازش باید حداقل 80% باشد
        }

    async def monitor_efficiency(self) -> dict:
        """
        پایش کارایی سیستم و ارائه پیشنهادات بهینه‌سازی.

        :return: دیکشنری شامل تحلیل عملکرد سیستم و راهکارهای بهبود.
        """
        # جمع‌آوری متریک‌های مربوط به پردازش‌های داده‌ای
        cpu_usage = await self.resource_monitor.get_cpu_usage()
        memory_usage = await self.resource_monitor.get_memory_usage()
        throughput = await self.metrics_collector.get_metric("throughput")

        # بررسی روند تغییر عملکرد
        efficiency_report = self._analyze_efficiency(cpu_usage, memory_usage, throughput)

        return efficiency_report

    def _analyze_efficiency(self, cpu_usage: list, memory_usage: list, throughput: list) -> dict:
        """
        تحلیل کارایی پردازش‌های سیستم و شناسایی مشکلات.

        :param cpu_usage: میزان مصرف CPU
        :param memory_usage: میزان مصرف حافظه
        :param throughput: توان عملیاتی پردازش‌ها
        :return: دیکشنری شامل پیشنهادات بهینه‌سازی
        """
        efficiency_issues = {}

        # تحلیل مصرف CPU و میزان بهره‌وری پردازش‌ها
        avg_cpu_usage = np.mean(cpu_usage) if cpu_usage else 0
        if avg_cpu_usage < self.efficiency_thresholds["cpu_efficiency"]:
            efficiency_issues["cpu_inefficiency"] = {
                "avg_usage": avg_cpu_usage,
                "suggestion": "افزایش تخصیص منابع پردازشی یا بهینه‌سازی کد"
            }

        # تحلیل مصرف حافظه و بهره‌وری آن
        avg_memory_usage = np.mean(memory_usage) if memory_usage else 0
        if avg_memory_usage > self.efficiency_thresholds["memory_efficiency"]:
            efficiency_issues["memory_overuse"] = {
                "avg_usage": avg_memory_usage,
                "suggestion": "افزایش استفاده از کش یا بهینه‌سازی حافظه"
            }

        # بررسی توان عملیاتی پردازش‌ها
        avg_throughput = np.mean(throughput) if throughput else 0
        if avg_throughput < self.efficiency_thresholds["throughput_efficiency"]:
            efficiency_issues["low_throughput"] = {
                "avg_throughput": avg_throughput,
                "suggestion": "افزایش پردازش موازی یا بهینه‌سازی الگوریتم پردازشی"
            }

        return efficiency_issues
