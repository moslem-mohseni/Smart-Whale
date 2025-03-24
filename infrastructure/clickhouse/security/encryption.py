# infrastructure/clickhouse/security/encryption.py
import base64
import os
import logging
from cryptography.fernet import Fernet, InvalidToken
from ..config.config import config

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    مدیریت رمزنگاری داده‌های حساس در ClickHouse

    از الگوریتم رمزنگاری متقارن Fernet برای رمزنگاری و رمزگشایی داده‌ها استفاده می‌کند.
    """

    def __init__(self, key: str = None):
        """
        مقداردهی اولیه سیستم رمزنگاری

        Args:
            key (str, optional): کلید رمزنگاری. اگر مقدار None باشد، از تنظیمات مرکزی استفاده می‌شود.
        """
        security_config = config.get_security_config()
        self.key = key or security_config["encryption_key"]

        # اعتبارسنجی کلید
        if not self.key:
            error_msg = "No encryption key provided. Encryption cannot function without a key."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # بررسی طول کلید و تطبیق با فرمت مورد نیاز
        try:
            # اگر کلید به فرمت base64 نباشد، تبدیل می‌کنیم
            if not self._is_valid_base64(self.key):
                logger.warning("Encryption key is not in valid base64 format. Attempting to encode.")
                self.key = base64.urlsafe_b64encode(self.key.encode()).decode()
                if len(self.key) < 32:
                    # پر کردن کلید تا رسیدن به طول مناسب
                    self.key = self.key.ljust(32, '=')

            # ایجاد شیء Fernet با کلید
            self.fernet = Fernet(self.key.encode())
            logger.info("Encryption manager initialized successfully.")
        except Exception as e:
            error_msg = f"Failed to initialize encryption with the provided key: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _is_valid_base64(self, key: str) -> bool:
        """
        بررسی معتبر بودن کلید در فرمت base64

        Args:
            key (str): کلید مورد بررسی

        Returns:
            bool: True اگر کلید در فرمت معتبر base64 باشد
        """
        try:
            # تلاش برای اینکه ببینیم آیا کلید به درستی در base64 رمزگذاری شده است
            decoded = base64.urlsafe_b64decode(key)
            return len(decoded) == 32
        except Exception:
            return False

    def encrypt(self, data: str) -> str:
        """
        رمزنگاری داده ورودی

        Args:
            data (str): داده متنی برای رمزنگاری

        Returns:
            str: داده رمزنگاری‌شده به فرمت base64

        Raises:
            ValueError: اگر داده ورودی خالی یا None باشد
        """
        if not data:
            raise ValueError("Cannot encrypt empty or None data")

        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise RuntimeError(f"Encryption failed: {str(e)}")

    def decrypt(self, encrypted_data: str) -> str:
        """
        رمزگشایی داده رمزنگاری‌شده

        Args:
            encrypted_data (str): داده رمزنگاری‌شده

        Returns:
            str: داده اصلی پس از رمزگشایی

        Raises:
            ValueError: اگر داده ورودی خالی یا None باشد
            InvalidToken: اگر داده رمزنگاری‌شده معتبر نباشد
        """
        if not encrypted_data:
            raise ValueError("Cannot decrypt empty or None data")

        try:
            decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except InvalidToken:
            logger.error("Invalid token provided for decryption")
            raise InvalidToken("The encrypted data is invalid or was encrypted with a different key")
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise RuntimeError(f"Decryption failed: {str(e)}")

    def rotate_key(self, new_key: str):
        """
        تغییر کلید رمزنگاری

        این متد به شما امکان می‌دهد کلید رمزنگاری را تغییر دهید. برای رمزگشایی داده‌های
        رمزنگاری‌شده با کلید قبلی، باید از همان کلید استفاده کنید.

        Args:
            new_key (str): کلید جدید
        """
        # ذخیره کلید قبلی
        old_fernet = self.fernet

        # تنظیم کلید جدید
        self.key = new_key
        self.fernet = Fernet(self.key.encode())

        logger.info("Encryption key rotated successfully.")

        return old_fernet
