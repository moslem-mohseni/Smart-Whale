from typing import Dict, Any, List
from collections import deque
import time
from .optimizer import ProcessingOptimizer

class TaskScheduler:
    """
    ماژول مدیریت زمان‌بندی وظایف پردازشی بر اساس اولویت و منابع در دسترس.
    """

    def __init__(self):
        """
        مقداردهی اولیه صف وظایف و تنظیمات مربوط به پردازش.
        """
        self.task_queue: deque = deque()  # صف وظایف پردازشی
        self.processing_optimizer = ProcessingOptimizer()

    def schedule_task(self, task: Dict[str, Any]):
        """
        اضافه کردن یک وظیفه به صف پردازش.
        :param task: اطلاعات وظیفه شامل پیچیدگی و میزان منابع موردنیاز.
        """
        self.task_queue.append(task)

    def execute_next_task(self) -> Dict[str, Any]:
        """
        اجرای وظیفه بعدی از صف پردازش.
        :return: نتیجه‌ی پردازش وظیفه‌ی اجرا شده.
        """
        if not self.task_queue:
            return {"status": "No tasks in queue"}

        task = self.task_queue.popleft()

        # بررسی منابع موجود و تخصیص بهینه
        allocated_resources = self.processing_optimizer.allocate_resources(task["complexity"])
        if not allocated_resources:
            return {"status": "Task postponed due to insufficient resources"}

        processing_time = self._get_processing_time(task["complexity"])
        time.sleep(processing_time)  # شبیه‌سازی پردازش

        return {
            "task_id": task.get("task_id", "unknown"),
            "status": "completed",
            "processing_time": processing_time
        }

    def _get_processing_time(self, complexity: str) -> float:
        """
        دریافت زمان تخمینی پردازش بر اساس سطح پیچیدگی.
        :param complexity: سطح پیچیدگی پردازش (quick, normal, deep).
        :return: زمان پردازش تخمینی.
        """
        processing_times = {
            "quick": 0.5,
            "normal": 1.0,
            "deep": 2.0
        }
        return processing_times.get(complexity, 1.0)

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        دریافت لیست وظایف در انتظار پردازش.
        :return: لیست وظایف در صف پردازش.
        """
        return list(self.task_queue)
