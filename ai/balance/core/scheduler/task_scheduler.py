import heapq
import time
from typing import Callable, Dict, Any, List, Tuple
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resilience.circuit_breaker.breaker_manager import BreakerManager
from core.monitoring.metrics.collector import MetricsCollector


class TaskScheduler:
    """
    زمان‌بندی وظایف پردازشی با بررسی منابع و مدیریت اولویت‌ها.
    """

    def __init__(self):
        self.task_queue: List[Tuple[float, int, Callable, Tuple, Dict[str, Any]]] = []
        self.task_counter = 0  # شمارنده برای حفظ ترتیب وظایف هم‌اولویت
        self.resource_monitor = ResourceMonitor()
        self.breaker_manager = BreakerManager()
        self.metrics_collector = MetricsCollector()

    def schedule_task(self, task: Callable, priority: int = 1, delay: float = 0, *args, **kwargs) -> None:
        """
        زمان‌بندی یک وظیفه برای اجرا در آینده.

        :param task: تابعی که باید اجرا شود.
        :param priority: اولویت اجرا (عدد کمتر = اولویت بالاتر).
        :param delay: مدت تأخیر قبل از اجرا (بر حسب ثانیه).
        :param args: آرگومان‌های ارسال‌شده به تابع.
        :param kwargs: آرگومان‌های کلیدی ارسال‌شده به تابع.
        """
        execution_time = time.time() + delay
        heapq.heappush(self.task_queue, (execution_time, priority, self.task_counter, task, args, kwargs))
        self.task_counter += 1

    def execute_next_task(self) -> bool:
        """
        اجرای وظیفه بعدی در صف با بررسی وضعیت منابع و قطع‌کننده مدار.

        :return: `True` اگر وظیفه‌ای اجرا شد، `False` اگر صف خالی بود یا منابع کافی نبود.
        """
        if not self.task_queue:
            return False

        execution_time, _, _, task, args, kwargs = heapq.heappop(self.task_queue)

        # اگر زمان اجرای آن هنوز نرسیده باشد، دوباره به صف بازگردانی می‌شود.
        if execution_time > time.time():
            heapq.heappush(self.task_queue, (execution_time, _, _, task, args, kwargs))
            return False

        # بررسی وضعیت منابع قبل از اجرا
        system_resources = self.resource_monitor.get_system_resources()
        if system_resources["cpu_usage"] > 0.85 or system_resources["memory_usage"] > 0.9:
            heapq.heappush(self.task_queue, (execution_time + 1, _, _, task, args, kwargs))
            return False  # اجرای وظیفه تا بهبود منابع تأخیر می‌افتد.

        # بررسی وضعیت قطع‌کننده مدار برای جلوگیری از پردازش‌های نامعتبر
        if self.breaker_manager.is_circuit_open():
            heapq.heappush(self.task_queue, (execution_time + 5, _, _, task, args, kwargs))
            return False

        # اجرای وظیفه
        start_time = time.time()
        task(*args, **kwargs)
        execution_time = time.time() - start_time

        # ثبت متریک اجرای وظیفه
        self.metrics_collector.collect_metric("task_execution_time", execution_time)

        return True

    def run_scheduler(self) -> None:
        """
        اجرای زمان‌بند برای پردازش وظایف در صف به صورت پیوسته.
        """
        while self.task_queue:
            if not self.execute_next_task():
                time.sleep(0.1)  # در صورت نبود وظیفه آماده، زمان‌بند کمی متوقف می‌شود

    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """
        دریافت لیست وظایف زمان‌بندی‌شده.

        :return: لیستی از وظایف در صف.
        """
        return [
            {"execution_time": task[0], "priority": task[1], "task_id": task[2]}
            for task in self.task_queue
        ]
