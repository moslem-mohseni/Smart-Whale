import numpy as np
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor


class PerformanceAnalyzer:
    """
    ماژولی برای تحلیل عملکرد پردازش‌های داده‌ای.
    این ماژول متریک‌های پردازشی را جمع‌آوری کرده و مشکلات عملکردی را شناسایی می‌کند.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
        self.performance_thresholds = {
            "execution_time": 0.75,  # اگر زمان اجرای پردازش بیش از 75% میانگین باشد، هشدار بدهد.
            "cpu_usage": 0.80,  # اگر مصرف CPU بیش از 80% باشد، هشدار بدهد.
            "memory_usage": 0.85  # اگر مصرف حافظه بیش از 85% باشد، هشدار بدهد.
        }

    async def analyze_performance(self) -> dict:
        """
        تحلیل عملکرد پردازش‌های جاری و شناسایی مشکلات.

        :return: دیکشنری شامل نتایج تحلیل و هشدارهای مربوطه.
        """
        # جمع‌آوری متریک‌های پردازشی
        execution_times = await self.metrics_collector.get_metric("execution_time")
        cpu_usages = await self.resource_monitor.get_cpu_usage()
        memory_usages = await self.resource_monitor.get_memory_usage()

        # تحلیل عملکرد پردازش‌ها
        performance_issues = self._detect_performance_issues(
            execution_times, cpu_usages, memory_usages
        )

        return performance_issues

    def _detect_performance_issues(self, execution_times: list, cpu_usages: list, memory_usages: list) -> dict:
        """
        شناسایی پردازش‌های ناکارآمد بر اساس متریک‌های جمع‌آوری‌شده.

        :param execution_times: لیست زمان‌های اجرای پردازش‌ها
        :param cpu_usages: لیست میزان مصرف CPU
        :param memory_usages: لیست میزان مصرف حافظه
        :return: دیکشنری شامل پردازش‌های دارای مشکل و هشدارهای مربوطه
        """
        issues = {}

        # میانگین زمان اجرای پردازش‌ها
        avg_execution_time = np.mean(execution_times) if execution_times else 0
        max_execution_time = np.percentile(execution_times, 90) if execution_times else 0

        # پردازش‌هایی که زمان اجرای بالایی دارند
        slow_processes = [
            time for time in execution_times if
            time > self.performance_thresholds["execution_time"] * avg_execution_time
        ]

        if slow_processes:
            issues["slow_processes"] = slow_processes

        # بررسی مصرف CPU
        high_cpu_usage = [
            usage for usage in cpu_usages if usage > self.performance_thresholds["cpu_usage"]
        ]

        if high_cpu_usage:
            issues["high_cpu_usage"] = high_cpu_usage

        # بررسی مصرف حافظه
        high_memory_usage = [
            usage for usage in memory_usages if usage > self.performance_thresholds["memory_usage"]
        ]

        if high_memory_usage:
            issues["high_memory_usage"] = high_memory_usage

        return issues
