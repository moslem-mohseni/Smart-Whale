import hashlib
from typing import Any


class IntegrityChecker:
    """
    بررسی صحت و یکپارچگی داده‌های منتقل‌شده با استفاده از هشینگ
    """

    @staticmethod
    def generate_checksum(data: Any) -> str:
        """
        تولید مقدار هش برای داده‌های ورودی جهت بررسی یکپارچگی
        :param data: داده‌های ورودی به‌صورت رشته یا بایت
        :return: مقدار هش به‌صورت رشته هگزادسیمال
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def verify_checksum(data: Any, expected_checksum: str) -> bool:
        """
        بررسی صحت داده‌ها با مقایسه مقدار هش محاسبه‌شده و مقدار مورد انتظار
        :param data: داده‌های ورودی به‌صورت رشته یا بایت
        :param expected_checksum: مقدار هش مورد انتظار برای تطابق
        :return: مقدار بولین که نشان می‌دهد داده صحیح است یا خیر
        """
        return IntegrityChecker.generate_checksum(data) == expected_checksum


# نمونه استفاده از IntegrityChecker برای تست
if __name__ == "__main__":
    checker = IntegrityChecker()
    sample_data = "This is a test string for integrity check."
    checksum = checker.generate_checksum(sample_data)
    print(f"Generated Checksum: {checksum}")
    print(f"Integrity Verified: {checker.verify_checksum(sample_data, checksum)}")
