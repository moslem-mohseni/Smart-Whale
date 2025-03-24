from typing import Dict, Any


class PrivacyGuard:
    """
    مکانیزم‌های حفاظت از حریم خصوصی هنگام اشتراک دانش بین مدل‌ها
    """

    def __init__(self):
        self.sensitive_keys = ["user_data", "private_info", "confidential"]  # کلیدهای حساس برای حذف

    def protect_privacy(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        حذف اطلاعات حساس از دانش قبل از اشتراک‌گذاری
        :param knowledge: داده‌های دانش ورودی
        :return: نسخه فیلترشده داده‌های دانش بدون اطلاعات حساس
        """
        return {key: value for key, value in knowledge.items() if key not in self.sensitive_keys}

    def add_sensitive_key(self, key: str) -> None:
        """
        اضافه کردن یک کلید جدید به لیست اطلاعات حساس برای حذف در آینده
        :param key: نام کلید حساس جدید
        """
        if key not in self.sensitive_keys:
            self.sensitive_keys.append(key)

    def remove_sensitive_key(self, key: str) -> None:
        """
        حذف یک کلید از لیست اطلاعات حساس
        :param key: نام کلید برای حذف
        """
        if key in self.sensitive_keys:
            self.sensitive_keys.remove(key)


# نمونه استفاده از PrivacyGuard برای تست
if __name__ == "__main__":
    guard = PrivacyGuard()
    sample_knowledge = {"accuracy": 0.95, "user_data": "Sensitive Info", "public_info": "General Data"}

    filtered_knowledge = guard.protect_privacy(sample_knowledge)
    print(f"Filtered Knowledge: {filtered_knowledge}")

    guard.add_sensitive_key("secret_key")
    print(f"Updated Sensitive Keys: {guard.sensitive_keys}")
