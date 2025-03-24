import hashlib
from typing import Dict


class SecurityManager:
    """
    مدیریت امنیتی برای جلوگیری از نشت اطلاعات و تضمین حفاظت داده‌ها در یادگیری فدراسیونی
    """

    def __init__(self):
        self.access_keys: Dict[str, str] = {}  # کلیدهای امنیتی برای احراز هویت

    def generate_access_key(self, user_id: str) -> str:
        """
        تولید کلید امنیتی برای یک کاربر خاص
        :param user_id: شناسه کاربر
        :return: کلید امنیتی هش‌شده
        """
        access_key = hashlib.sha256(user_id.encode()).hexdigest()
        self.access_keys[user_id] = access_key
        return access_key

    def verify_access_key(self, user_id: str, access_key: str) -> bool:
        """
        بررسی اعتبار کلید امنیتی کاربر
        :param user_id: شناسه کاربر
        :param access_key: کلید ارائه‌شده توسط کاربر
        :return: مقدار بولین که نشان‌دهنده معتبر بودن یا نبودن کلید است
        """
        return self.access_keys.get(user_id) == access_key

    def revoke_access(self, user_id: str) -> None:
        """
        لغو دسترسی یک کاربر با حذف کلید امنیتی وی
        :param user_id: شناسه کاربر
        """
        if user_id in self.access_keys:
            del self.access_keys[user_id]


# نمونه استفاده از SecurityManager برای تست
if __name__ == "__main__":
    security = SecurityManager()
    user_id = "user123"
    key = security.generate_access_key(user_id)
    print(f"Generated Key for {user_id}: {key}")
    print(f"Verification Result: {security.verify_access_key(user_id, key)}")
    security.revoke_access(user_id)
    print(f"Verification after Revocation: {security.verify_access_key(user_id, key)}")
