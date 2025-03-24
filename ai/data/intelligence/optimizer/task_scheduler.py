import heapq
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor

class TaskScheduler:
    """
    ماژولی برای زمان‌بندی و اولویت‌بندی پردازش‌های داده‌ای.
    این ماژول پردازش‌های مختلف را بر اساس میزان اهمیت و ظرفیت منابع سیستم، مدیریت و زمان‌بندی می‌کند.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
        self.task_queue = []  # استفاده از یک heap برای مدیریت پردازش‌ها بر اساس اولویت

    async def schedule_tasks(self, tasks: list) -> list:
        """
        مدیریت و زمان‌بندی پردازش‌های داده‌ای.

        :param tasks: لیستی از پردازش‌های داده‌ای که باید اجرا شوند.
        :return: لیستی از پردازش‌ها به ترتیب اجرا.
        """
        if not tasks:
            return {"status": "no_tasks", "message": "هیچ پردازشی برای زمان‌بندی وجود ندارد."}

        # جمع‌آوری متریک‌های مربوط به منابع
        cpu_usage = await self.resource_monitor.get_cpu_usage()
        memory_usage = await self.resource_monitor.get_memory_usage()

        # اولویت‌بندی پردازش‌ها
        scheduled_tasks = self._prioritize_tasks(tasks, cpu_usage, memory_usage)

        return scheduled_tasks

    def _prioritize_tasks(self, tasks: list, cpu_usage: list, memory_usage: list) -> list:
        """
        زمان‌بندی پردازش‌ها با در نظر گرفتن اولویت‌ها.

        :param tasks: لیستی از پردازش‌ها
        :param cpu_usage: میزان مصرف CPU فعلی
        :param memory_usage: میزان مصرف RAM فعلی
        :return: لیستی از پردازش‌ها بر اساس ترتیب اجرا
        """
        prioritized_tasks = []

        for task in tasks:
            # محاسبه اولویت پردازش (عدد کوچکتر = اولویت بالاتر)
            priority = self._calculate_priority(task, cpu_usage, memory_usage)
            heapq.heappush(self.task_queue, (priority, task))

        # ترتیب اجرای پردازش‌ها بر اساس اولویت
        while self.task_queue:
            _, task = heapq.heappop(self.task_queue)
            prioritized_tasks.append(task)

        return prioritized_tasks

    def _calculate_priority(self, task: dict, cpu_usage: list, memory_usage: list) -> int:
        """
        محاسبه اولویت پردازش با در نظر گرفتن منابع مصرفی و اهمیت پردازش.

        :param task: اطلاعات پردازش (شامل سطح اهمیت و نیازمندی‌های منابع)
        :param cpu_usage: میزان مصرف CPU
        :param memory_usage: میزان مصرف RAM
        :return: عدد اولویت (عدد کوچکتر = پردازش اولویت بالاتر دارد)
        """
        base_priority = task.get("priority", 5)  # مقدار پیش‌فرض اولویت (۱=بالاترین، ۱۰=کمترین)

        # افزایش اولویت پردازش‌هایی که نیازمند منابع کمتری هستند
        if task.get("cpu_demand", 0) < 0.2 * max(cpu_usage, default=1):
            base_priority -= 1
        if task.get("memory_demand", 0) < 0.2 * max(memory_usage, default=1):
            base_priority -= 1

        return max(1, base_priority)  # حداقل مقدار اولویت ۱ است
