from typing import Dict, Any, List
import hashlib

class PrivacyGuard:
    """
    ماژول مدیریت امنیت و حریم خصوصی در اشتراک‌گذاری دانش بین مدل‌های فدراسیونی.
    """

    def __init__(self):
        """
        مقداردهی اولیه لیست داده‌های حساس و سطح امنیتی.
        """
        self.sensitive_keys: List[str] = ["user_id", "email", "phone_number", "ip_address"]
        self.encryption_salt = "smartwhale_salt"

    def protect_privacy(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        پردازش و حذف داده‌های حساس از دانش مدل.
        :param knowledge: دیکشنری شامل دانش مدل.
        :return: نسخه پاک‌سازی‌شده از دانش.
        """
        protected_knowledge = {}

        for key, value in knowledge.items():
            if key in self.sensitive_keys:
                protected_knowledge[key] = self._hash_sensitive_data(value)
            else:
                protected_knowledge[key] = value

        return protected_knowledge

    def _hash_sensitive_data(self, data: Any) -> str:
        """
        رمزنگاری و هش کردن داده‌های حساس برای حفظ امنیت.
        :param data: مقدار داده‌ی حساس.
        :return: هش رمزنگاری‌شده‌ی داده.
        """
        data_str = str(data) + self.encryption_salt
        return hashlib.sha256(data_str.encode()).hexdigest()

    def add_sensitive_key(self, key: str) -> None:
        """
        اضافه کردن یک کلید جدید به لیست داده‌های حساس.
        :param key: نام کلید حساس جدید.
        """
        if key not in self.sensitive_keys:
            self.sensitive_keys.append(key)

    def remove_sensitive_key(self, key: str) -> None:
        """
        حذف یک کلید از لیست داده‌های حساس.
        :param key: نام کلید حساس.
        """
        if key in self.sensitive_keys:
            self.sensitive_keys.remove(key)

    def get_sensitive_keys(self) -> List[str]:
        """
        دریافت لیست کلیدهای حساس.
        :return: لیست کلیدهای حساس.
        """
        return self.sensitive_keys
