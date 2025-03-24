from typing import Dict, Any, List


class FederatedLearner:
    """
    مدیریت یادگیری فدراسیونی بین مدل‌های هوش مصنوعی برای هماهنگ‌سازی دانش بدون افشای داده‌های خام
    """

    def __init__(self):
        self.local_updates: Dict[str, List[float]] = {}  # نگهداری وزن‌های محلی مدل‌ها

    def collect_model_update(self, model_name: str, update: List[float]) -> None:
        """
        دریافت و ذخیره به‌روزرسانی مدل از یک مدل محلی
        :param model_name: نام مدل
        :param update: لیست وزن‌های به‌روزرسانی شده
        """
        if model_name not in self.local_updates:
            self.local_updates[model_name] = []
        self.local_updates[model_name].append(update)

    def get_updates(self) -> Dict[str, List[List[float]]]:
        """
        دریافت تمامی به‌روزرسانی‌های جمع‌آوری‌شده از مدل‌های مختلف
        :return: دیکشنری شامل به‌روزرسانی‌های تمام مدل‌ها
        """
        return self.local_updates

    def clear_updates(self) -> None:
        """
        پاک‌سازی داده‌های به‌روزرسانی برای شروع یک دوره جدید یادگیری فدراسیونی
        """
        self.local_updates.clear()


# نمونه استفاده از FederatedLearner برای تست
if __name__ == "__main__":
    learner = FederatedLearner()
    learner.collect_model_update("model_a", [0.1, 0.2, 0.3])
    learner.collect_model_update("model_b", [0.4, 0.5, 0.6])

    print(f"Collected Updates: {learner.get_updates()}")
    learner.clear_updates()
    print(f"Updates after clearing: {learner.get_updates()}")
