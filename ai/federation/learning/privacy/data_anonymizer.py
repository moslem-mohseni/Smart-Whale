from typing import Dict
import hashlib


class DataAnonymizer:
    """
    ناشناس‌سازی داده‌ها برای حفظ حریم خصوصی کاربران در یادگیری فدراسیونی
    """

    def __init__(self):
        self.salt = "secure_salt"  # مقدار تصادفی برای هش کردن داده‌ها

    def anonymize_data(self, data: Dict[str, str]) -> Dict[str, str]:
        """
        تبدیل داده‌های حساس به مقادیر هش‌شده برای حفظ حریم خصوصی
        :param data: دیکشنری شامل داده‌های حساس
        :return: دیکشنری شامل داده‌های ناشناس‌شده
        """
        return {key: hashlib.sha256((value + self.salt).encode()).hexdigest() for key, value in data.items()}

    def set_salt(self, new_salt: str) -> None:
        """
        تنظیم مقدار جدید salt برای افزایش امنیت هش‌ها
        :param new_salt: مقدار جدید salt
        """
        self.salt = new_salt


# نمونه استفاده از DataAnonymizer برای تست
if __name__ == "__main__":
    anonymizer = DataAnonymizer()
    sensitive_data = {"user_id": "12345", "email": "user@example.com"}
    anonymized = anonymizer.anonymize_data(sensitive_data)
    print(f"Original Data: {sensitive_data}")
    print(f"Anonymized Data: {anonymized}")
