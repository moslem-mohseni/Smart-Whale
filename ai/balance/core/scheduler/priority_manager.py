from typing import Dict, List, Tuple
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor

class PriorityManager:
    """
    مدیریت اولویت وظایف برای بهینه‌سازی پردازش‌ها.
    """

    def __init__(self):
        """
        مقداردهی اولیه اولویت وظایف. وظایف با مقدار کمتر اولویت بالاتری دارند.
        """
        self.task_priorities: Dict[str, int] = {}
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()

    def set_priority(self, task_id: str, priority: int) -> None:
        """
        تعیین اولویت یک وظیفه.

        :param task_id: شناسه وظیفه
        :param priority: سطح اولویت (عدد کمتر = اولویت بالاتر)
        """
        self.task_priorities[task_id] = priority
        self.metrics_collector.collect_metric("task_priority_set", {"task_id": task_id, "priority": priority})

    def get_priority(self, task_id: str) -> int:
        """
        دریافت سطح اولویت یک وظیفه.

        :param task_id: شناسه وظیفه
        :return: مقدار اولویت (پیش‌فرض ۵ در صورت عدم تنظیم)
        """
        return self.task_priorities.get(task_id, 5)

    def adjust_priority(self, task_id: str, adjustment: int) -> None:
        """
        تغییر سطح اولویت یک وظیفه بر اساس مصرف منابع.

        :param task_id: شناسه وظیفه
        :param adjustment: مقدار تغییر در اولویت (مثلاً +۱ برای کاهش اولویت)
        """
        system_resources = self.resource_monitor.get_system_resources()

        # اگر سیستم تحت فشار است، اولویت را کاهش دهیم
        if system_resources["cpu_usage"] > 0.85 or system_resources["memory_usage"] > 0.9:
            adjustment += 1  # افزایش مقدار عددی یعنی کاهش اولویت

        if task_id in self.task_priorities:
            self.task_priorities[task_id] = max(1, self.task_priorities[task_id] + adjustment)

        self.metrics_collector.collect_metric("task_priority_adjusted", {"task_id": task_id, "adjustment": adjustment})

    def get_sorted_tasks(self) -> List[Tuple[str, int]]:
        """
        دریافت لیست وظایف مرتب‌شده بر اساس اولویت.

        :return: لیستی از وظایف به همراه سطح اولویت، مرتب‌شده از کمترین به بیشترین مقدار
        """
        return sorted(self.task_priorities.items(), key=lambda item: item[1])
