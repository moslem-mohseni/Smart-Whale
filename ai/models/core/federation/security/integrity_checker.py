import hashlib
from typing import Dict, Any

class IntegrityChecker:
    """
    ماژول بررسی صحت و یکپارچگی داده‌های فدراسیونی برای جلوگیری از تغییرات ناخواسته.
    """

    def __init__(self):
        """
        مقداردهی اولیه.
        """
        self.data_checksums: Dict[str, str] = {}

    def generate_checksum(self, data: Any) -> str:
        """
        تولید `checksum` برای داده ورودی.
        :param data: داده‌ای که باید بررسی شود.
        :return: مقدار هش تولید شده از داده.
        """
        data_str = str(data).encode()
        return hashlib.sha256(data_str).hexdigest()

    def store_checksum(self, data_id: str, data: Any):
        """
        ذخیره `checksum` برای یک داده‌ی مشخص.
        :param data_id: شناسه داده.
        :param data: مقدار داده.
        """
        self.data_checksums[data_id] = self.generate_checksum(data)

    def verify_integrity(self, data_id: str, new_data: Any) -> bool:
        """
        بررسی صحت و یکپارچگی داده با مقایسه `checksum` قبلی و جدید.
        :param data_id: شناسه داده.
        :param new_data: مقدار داده‌ی جدید برای مقایسه.
        :return: `True` اگر داده تغییری نکرده باشد، `False` در غیر اینصورت.
        """
        if data_id not in self.data_checksums:
            return False  # داده‌ای برای مقایسه موجود نیست

        new_checksum = self.generate_checksum(new_data)
        return new_checksum == self.data_checksums[data_id]

    def remove_checksum(self, data_id: str):
        """
        حذف `checksum` ذخیره‌شده برای یک داده.
        :param data_id: شناسه داده.
        """
        if data_id in self.data_checksums:
            del self.data_checksums[data_id]

    def get_all_checksums(self) -> Dict[str, str]:
        """
        دریافت تمامی `checksum`‌های ذخیره‌شده.
        :return: دیکشنری شامل شناسه داده‌ها و `checksum` آن‌ها.
        """
        return self.data_checksums
