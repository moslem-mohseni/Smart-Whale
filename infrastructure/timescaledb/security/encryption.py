import os
import base64
import logging
from typing import Optional
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class EncryptionManager:
    """مدیریت رمزنگاری داده‌های حساس در TimescaleDB"""

    def __init__(self):
        """مقداردهی اولیه و بارگذاری کلید رمزنگاری"""
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("❌ کلید رمزنگاری در `.env` تعریف نشده است.")
        self.fernet = Fernet(encryption_key)

    def encrypt(self, plain_text: str) -> str:
        """
        رمزنگاری مقدار داده‌شده

        Args:
            plain_text (str): مقدار متنی برای رمزنگاری

        Returns:
            str: مقدار رمزنگاری‌شده (Base64)
        """
        encrypted_text = self.fernet.encrypt(plain_text.encode())
        return base64.urlsafe_b64encode(encrypted_text).decode()

    def decrypt(self, encrypted_text: str) -> Optional[str]:
        """
        رمزگشایی مقدار رمزنگاری‌شده

        Args:
            encrypted_text (str): مقدار رمزنگاری‌شده (Base64)

        Returns:
            Optional[str]: مقدار رمزگشایی‌شده یا None در صورت خطا
        """
        try:
            decrypted_text = self.fernet.decrypt(base64.urlsafe_b64decode(encrypted_text)).decode()
            return decrypted_text
        except Exception as e:
            logger.error(f"❌ خطا در رمزگشایی داده: {e}")
            return None
