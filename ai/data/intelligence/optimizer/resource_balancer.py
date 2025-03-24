import numpy as np
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor

class ResourceBalancer:
    """
    ماژولی برای متعادل‌سازی منابع پردازشی (CPU، RAM، I/O) بین فرآیندهای مختلف.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
        self.balance_thresholds = {
            "cpu_usage_limit": 0.80,  # اگر مصرف CPU بیش از 80% باشد، باید متعادل‌سازی شود.
            "memory_usage_limit": 0.85,  # اگر مصرف RAM بیش از 85% باشد، نیاز به تنظیم مجدد منابع داریم.
            "io_usage_limit": 0.75  # اگر مصرف I/O دیسک از 75% عبور کند، باید فرآیندها را بهینه کنیم.
        }

    async def balance_resources(self) -> dict:
        """
        بررسی و متعادل‌سازی مصرف منابع پردازشی.

        :return: دیکشنری شامل پیشنهادات متعادل‌سازی.
        """
        # جمع‌آوری متریک‌های مصرف منابع
        cpu_usage = await self.resource_monitor.get_cpu_usage()
        memory_usage = await self.resource_monitor.get_memory_usage()
        io_usage = await self.metrics_collector.get_metric("io_usage")

        # تحلیل و توزیع منابع بین پردازش‌ها
        balance_report = self._analyze_resource_distribution(cpu_usage, memory_usage, io_usage)

        return balance_report

    def _analyze_resource_distribution(self, cpu_usage: list, memory_usage: list, io_usage: list) -> dict:
        """
        تحلیل توزیع منابع و ارائه راهکارهای متعادل‌سازی.

        :param cpu_usage: میزان مصرف CPU توسط پردازش‌ها.
        :param memory_usage: میزان مصرف RAM توسط پردازش‌ها.
        :param io_usage: میزان مصرف دیسک توسط پردازش‌ها.
        :return: دیکشنری شامل پیشنهادات متعادل‌سازی منابع.
        """
        balance_suggestions = {}

        # بررسی مصرف CPU و توزیع بهتر منابع
        avg_cpu_usage = np.mean(cpu_usage) if cpu_usage else 0
        if avg_cpu_usage > self.balance_thresholds["cpu_usage_limit"]:
            balance_suggestions["cpu_balance"] = {
                "avg_usage": avg_cpu_usage,
                "suggestion": "کاهش تعداد پردازش‌های همزمان یا افزایش تخصیص هسته‌های CPU."
            }

        # بررسی مصرف RAM و تنظیم تخصیص حافظه
        avg_memory_usage = np.mean(memory_usage) if memory_usage else 0
        if avg_memory_usage > self.balance_thresholds["memory_usage_limit"]:
            balance_suggestions["memory_balance"] = {
                "avg_usage": avg_memory_usage,
                "suggestion": "بهینه‌سازی مصرف حافظه یا تخصیص حافظه بیشتر برای فرآیندهای مهم‌تر."
            }

        # بررسی مصرف I/O دیسک و جلوگیری از سربار
        avg_io_usage = np.mean(io_usage) if io_usage else 0
        if avg_io_usage > self.balance_thresholds["io_usage_limit"]:
            balance_suggestions["io_balance"] = {
                "avg_usage": avg_io_usage,
                "suggestion": "بهینه‌سازی فرآیندهای دیسک یا توزیع داده‌ها بین دیسک‌های مختلف."
            }

        return balance_suggestions
