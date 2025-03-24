from typing import Dict, Any, List


class ModelCoordinator:
    """
    هماهنگ‌کننده تعاملات بین مدل‌های هوش مصنوعی برای بهینه‌سازی پردازش
    """

    def __init__(self):
        self.model_status: Dict[str, str] = {}  # وضعیت مدل‌ها (idle, busy)

    def register_model(self, model_name: str) -> None:
        """
        ثبت یک مدل جدید در سیستم هماهنگ‌سازی
        :param model_name: نام مدل جدید
        """
        self.model_status[model_name] = "idle"

    def assign_task(self, model_name: str, task: Dict[str, Any]) -> bool:
        """
        تخصیص یک وظیفه به مدل مشخص‌شده در صورت در دسترس بودن
        :param model_name: نام مدل
        :param task: جزئیات وظیفه
        :return: مقدار بولین که نشان می‌دهد وظیفه تخصیص یافت یا نه
        """
        if self.model_status.get(model_name) == "idle":
            self.model_status[model_name] = "busy"
            # اجرای پردازش در مدل (شبیه‌سازی‌شده)
            print(f"Task assigned to {model_name}: {task}")
            return True
        return False

    def release_model(self, model_name: str) -> None:
        """
        آزادسازی مدل پس از اتمام پردازش
        :param model_name: نام مدل آزادشده
        """
        if model_name in self.model_status:
            self.model_status[model_name] = "idle"

    def get_available_models(self) -> List[str]:
        """
        دریافت لیست مدل‌های آماده برای پردازش
        :return: لیستی از نام مدل‌های آزاد
        """
        return [model for model, status in self.model_status.items() if status == "idle"]


# نمونه استفاده از ModelCoordinator برای تست
if __name__ == "__main__":
    coordinator = ModelCoordinator()
    coordinator.register_model("model_a")
    coordinator.register_model("model_b")

    task = {"type": "classification", "data": "sample input"}
    assigned = coordinator.assign_task("model_a", task)
    print(f"Task assigned to model_a: {assigned}")
    print(f"Available Models: {coordinator.get_available_models()}")
    coordinator.release_model("model_a")
    print(f"Available Models after release: {coordinator.get_available_models()}")
