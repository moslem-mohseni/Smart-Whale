import numpy as np
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor

class WorkloadBalancer:
    """
    ماژولی برای متعادل‌سازی بار پردازشی بین پردازش‌های مختلف.
    این ماژول منابع را بین پردازش‌های سنگین و سبک توزیع می‌کند تا سیستم کارایی بهتری داشته باشد.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
        self.balance_thresholds = {
            "cpu_usage_limit": 0.85,  # اگر مصرف CPU بیش از 85% باشد، نیاز به توزیع مجدد بار داریم.
            "memory_usage_limit": 0.80,  # اگر مصرف RAM بیش از 80% باشد، باید پردازش‌های سنگین را متعادل کنیم.
            "max_task_load": 0.75  # اگر یک پردازش بیش از 75% منابع را مصرف کند، بار را بین پردازش‌های دیگر توزیع می‌کنیم.
        }

    async def balance_workload(self, processes: dict) -> dict:
        """
        بررسی و متعادل‌سازی بار پردازشی بین پردازش‌های مختلف.

        :param processes: دیکشنری شامل پردازش‌ها و میزان مصرف منابع آن‌ها.
        :return: دیکشنری شامل پیشنهادات توزیع بار.
        """
        if not processes:
            return {"status": "no_processes", "message": "هیچ پردازشی برای متعادل‌سازی وجود ندارد."}

        # جمع‌آوری متریک‌های مصرف منابع
        cpu_usage = await self.resource_monitor.get_cpu_usage()
        memory_usage = await self.resource_monitor.get_memory_usage()

        # تحلیل و توزیع بار پردازشی
        balance_report = self._analyze_workload_distribution(processes, cpu_usage, memory_usage)

        return balance_report

    def _analyze_workload_distribution(self, processes: dict, cpu_usage: list, memory_usage: list) -> dict:
        """
        تحلیل بار پردازشی و ارائه راهکارهای متعادل‌سازی.

        :param processes: دیکشنری شامل پردازش‌ها و میزان مصرف منابع آن‌ها.
        :param cpu_usage: میزان مصرف CPU
        :param memory_usage: میزان مصرف RAM
        :return: دیکشنری شامل پیشنهادات متعادل‌سازی.
        """
        balance_suggestions = {}

        # بررسی مصرف CPU و توزیع بهتر پردازش‌ها
        avg_cpu_usage = np.mean(cpu_usage) if cpu_usage else 0
        if avg_cpu_usage > self.balance_thresholds["cpu_usage_limit"]:
            balance_suggestions["cpu_balance"] = {
                "avg_usage": avg_cpu_usage,
                "suggestion": "انتقال پردازش‌های سنگین‌تر به منابع کمتر استفاده‌شده."
            }

        # بررسی مصرف RAM و تنظیم تخصیص حافظه
        avg_memory_usage = np.mean(memory_usage) if memory_usage else 0
        if avg_memory_usage > self.balance_thresholds["memory_usage_limit"]:
            balance_suggestions["memory_balance"] = {
                "avg_usage": avg_memory_usage,
                "suggestion": "افزایش استفاده از کش یا توزیع مجدد پردازش‌های پرمصرف به منابع دیگر."
            }

        # بررسی پردازش‌هایی که بیش از حد از منابع استفاده می‌کنند
        for process, usage in processes.items():
            if usage["load"] > self.balance_thresholds["max_task_load"]:
                balance_suggestions[f"reduce_load_{process}"] = {
                    "current_load": usage["load"],
                    "suggestion": "توزیع بخشی از پردازش به دیگر منابع پردازشی."
                }

        return balance_suggestions
