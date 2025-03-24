from typing import Dict, Any


class PriorityHandler:
    """
    این کلاس مسئول مدیریت و تنظیم اولویت دسته‌های پردازشی است.
    """

    def __init__(self):
        """
        مقداردهی اولیه با سطح‌های پیش‌فرض اولویت.
        """
        self.priority_levels = {
            "high": 1,
            "medium": 2,
            "low": 3
        }

    def assign_priority(self, batch_data: Dict[str, Any]) -> int:
        """
        تعیین سطح اولویت برای دسته بر اساس نوع درخواست.
        """
        priority_key = batch_data.get("priority", "medium")
        return self.priority_levels.get(priority_key, 2)  # مقدار پیش‌فرض medium است.

    def adjust_priority(self, batch_id: str, new_priority: str) -> int:
        """
        تنظیم اولویت جدید برای یک دسته خاص.
        """
        return self.priority_levels.get(new_priority, 2)
