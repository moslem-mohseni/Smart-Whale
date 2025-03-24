from typing import Dict


class VersionController:
    """
    مدیریت نسخه‌های مختلف دانش برای ردیابی تغییرات و بازگردانی نسخه‌ها
    """

    def __init__(self):
        self.version_history: Dict[str, list] = {}  # نگهداری تاریخچه نسخه‌های مدل‌ها

    def add_version(self, model_name: str, version: str) -> None:
        """
        افزودن نسخه جدید به تاریخچه مدل مشخص شده
        :param model_name: نام مدل
        :param version: نسخه جدید برای ثبت
        """
        if model_name not in self.version_history:
            self.version_history[model_name] = []
        self.version_history[model_name].append(version)

    def get_latest_version(self, model_name: str) -> str:
        """
        دریافت آخرین نسخه ذخیره‌شده یک مدل
        :param model_name: نام مدل
        :return: آخرین نسخه مدل یا مقدار پیش‌فرض "unknown"
        """
        return self.version_history.get(model_name, ["unknown"])[-1]

    def rollback_version(self, model_name: str, steps: int = 1) -> str:
        """
        بازگردانی به نسخه قبلی مدل مشخص شده
        :param model_name: نام مدل
        :param steps: تعداد گام‌هایی که باید به عقب برگردد
        :return: نسخه بازگردانی‌شده یا مقدار "unknown" در صورت نبود نسخه قبلی
        """
        if model_name not in self.version_history or len(self.version_history[model_name]) < steps:
            return "unknown"

        rollback_index = max(0, len(self.version_history[model_name]) - steps - 1)
        return self.version_history[model_name][rollback_index]


# نمونه استفاده از VersionController برای تست
if __name__ == "__main__":
    version_controller = VersionController()
    version_controller.add_version("model_a", "v1.0")
    version_controller.add_version("model_a", "v1.1")
    version_controller.add_version("model_a", "v1.2")

    print(f"Latest Version: {version_controller.get_latest_version('model_a')}")
    print(f"Rollback to Previous Version: {version_controller.rollback_version('model_a', 1)}")
