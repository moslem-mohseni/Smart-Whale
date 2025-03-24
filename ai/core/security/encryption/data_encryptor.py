import base64
import logging
from cryptography.fernet import Fernet

class DataEncryptor:
    def __init__(self, secret_key: str = None):
        """
        مدیریت رمزنگاری داده‌ها با استفاده از AES-256
        :param secret_key: کلید رمزنگاری (در صورت عدم ارسال، کلید جدید ایجاد می‌شود)
        """
        self.logger = logging.getLogger("DataEncryptor")

        if secret_key:
            self.secret_key = secret_key.encode()
        else:
            self.secret_key = Fernet.generate_key()
            self.logger.warning("⚠️ کلید رمزنگاری جدید ایجاد شد. برای امنیت بهتر، کلید را ذخیره کنید!")

        self.cipher = Fernet(self.secret_key)

    def encrypt(self, data: str) -> str:
        """ رمزنگاری داده‌های ورودی """
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """ رمزگشایی داده‌های رمزنگاری‌شده """
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            return self.cipher.decrypt(decoded_data).decode()
        except Exception as e:
            self.logger.error(f"❌ خطا در رمزگشایی داده: {e}")
            return None
