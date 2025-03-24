from typing import Dict, Any, List, Optional
from collections import deque
from .model_coordinator import ModelCoordinator

class TaskManager:
    """
    ماژول مدیریت وظایف پردازشی و توزیع هوشمند وظایف بین مدل‌های فدراسیونی.
    """

    def __init__(self):
        """
        مقداردهی اولیه با نگهداری صف وظایف و هماهنگی با `ModelCoordinator`.
        """
        self.task_queue: deque = deque()  # صف وظایف در انتظار پردازش
        self.model_coordinator = ModelCoordinator()

    def add_task(self, task: Dict[str, Any]):
        """
        اضافه کردن یک وظیفه‌ی جدید به صف پردازش.
        :param task: اطلاعات وظیفه شامل نوع پردازش و اولویت.
        """
        self.task_queue.append(task)

    def assign_task(self) -> Optional[Dict[str, Any]]:
        """
        تخصیص وظیفه به یک مدل مناسب و حذف آن از صف پردازش.
        :return: وظیفه‌ی اختصاص‌یافته یا `None` در صورت خالی بودن صف.
        """
        if not self.task_queue:
            return None  # صف خالی است

        task = self.task_queue.popleft()
        available_models = self.model_coordinator.get_all_models()

        if not available_models:
            self.task_queue.appendleft(task)  # بازگرداندن وظیفه به صف
            return None  # هیچ مدلی برای پردازش در دسترس نیست

        # انتخاب اولین مدل در دسترس (می‌توان بهینه‌تر کرد)
        selected_model = available_models[0]
        if self.model_coordinator.assign_task(selected_model, task):
            return {"model": selected_model, "task": task}

        return None  # تخصیص ناموفق

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        دریافت لیست وظایف در انتظار پردازش.
        :return: لیست وظایف در صف.
        """
        return list(self.task_queue)

    def complete_task(self, model_id: str, task_id: str) -> bool:
        """
        حذف یک وظیفه از صف و ثبت اتمام پردازش آن در `ModelCoordinator`.
        :param model_id: شناسه مدل پردازنده‌ی وظیفه.
        :param task_id: شناسه وظیفه‌ای که باید تکمیل شود.
        :return: `True` در صورت موفقیت و `False` در صورت عدم موفقیت.
        """
        return self.model_coordinator.complete_task(model_id, task_id)
