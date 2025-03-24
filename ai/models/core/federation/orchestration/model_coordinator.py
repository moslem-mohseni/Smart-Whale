from typing import Dict, Any, List, Optional
from collections import defaultdict

class ModelCoordinator:
    """
    ماژول مدیریت هماهنگی بین مدل‌های فدراسیونی.
    """

    def __init__(self):
        """
        مقداردهی اولیه و نگهداری اطلاعات مدل‌های ثبت‌شده و وظایف اختصاص‌یافته.
        """
        self.registered_models: List[str] = []  # فهرست مدل‌های ثبت‌شده
        self.model_tasks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # مدل -> لیست وظایف

    def register_model(self, model_id: str):
        """
        ثبت یک مدل جدید در سیستم فدراسیونی.
        :param model_id: شناسه مدل.
        """
        if model_id not in self.registered_models:
            self.registered_models.append(model_id)

    def deregister_model(self, model_id: str):
        """
        حذف یک مدل از سیستم فدراسیونی.
        :param model_id: شناسه مدل.
        """
        if model_id in self.registered_models:
            self.registered_models.remove(model_id)
            if model_id in self.model_tasks:
                del self.model_tasks[model_id]  # حذف تمام وظایف مربوط به این مدل

    def assign_task(self, model_id: str, task: Dict[str, Any]) -> bool:
        """
        اختصاص وظیفه‌ی پردازشی به یک مدل خاص.
        :param model_id: شناسه مدل.
        :param task: اطلاعات مربوط به وظیفه.
        :return: `True` در صورت موفقیت و `False` در صورت عدم موفقیت.
        """
        if model_id in self.registered_models:
            self.model_tasks[model_id].append(task)
            return True
        return False  # مدل موردنظر ثبت نشده است

    def get_tasks(self, model_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        دریافت لیست وظایف تخصیص‌یافته به یک مدل.
        :param model_id: شناسه مدل.
        :return: لیست وظایف یا `None` اگر مدل ثبت نشده باشد.
        """
        return self.model_tasks.get(model_id, None)

    def complete_task(self, model_id: str, task_id: str) -> bool:
        """
        حذف یک وظیفه از لیست پردازش‌های یک مدل پس از تکمیل آن.
        :param model_id: شناسه مدل.
        :param task_id: شناسه وظیفه‌ای که باید حذف شود.
        :return: `True` در صورت حذف موفقیت‌آمیز و `False` در صورت عدم موفقیت.
        """
        if model_id in self.model_tasks:
            self.model_tasks[model_id] = [task for task in self.model_tasks[model_id] if task["task_id"] != task_id]
            return True
        return False  # مدل موردنظر وظایفی ندارد

    def get_all_models(self) -> List[str]:
        """
        دریافت فهرست تمام مدل‌های ثبت‌شده.
        :return: لیست شناسه‌های مدل‌های فدراسیونی.
        """
        return self.registered_models
