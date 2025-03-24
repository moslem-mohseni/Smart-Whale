from typing import Dict, Any, List


class TaskCoordinator:
    """
    هماهنگ‌کننده توزیع وظایف بین مدل‌های هوش مصنوعی برای بهینه‌سازی پردازش
    """

    def __init__(self):
        self.task_queue: List[Dict[str, Any]] = []  # صف وظایف در انتظار پردازش

    def add_task(self, task: Dict[str, Any]) -> None:
        """
        اضافه کردن یک وظیفه جدید به صف پردازش
        :param task: جزئیات وظیفه شامل نوع و داده‌های مورد نیاز
        """
        self.task_queue.append(task)

    def assign_task(self) -> Dict[str, Any]:
        """
        اختصاص وظیفه بعدی از صف به مدل‌های در دسترس
        :return: وظیفه‌ای که برای پردازش ارسال شده یا مقدار خالی در صورت نبود وظیفه
        """
        if self.task_queue:
            return self.task_queue.pop(0)
        return {}

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        دریافت لیست وظایف در انتظار پردازش
        :return: لیست وظایف در صف پردازش
        """
        return self.task_queue


# نمونه استفاده از TaskCoordinator برای تست
if __name__ == "__main__":
    coordinator = TaskCoordinator()
    coordinator.add_task({"type": "classification", "data": "sample input 1"})
    coordinator.add_task({"type": "segmentation", "data": "sample input 2"})

    print(f"Pending Tasks: {coordinator.get_pending_tasks()}")
    assigned_task = coordinator.assign_task()
    print(f"Assigned Task: {assigned_task}")
    print(f"Pending Tasks after assignment: {coordinator.get_pending_tasks()}")
