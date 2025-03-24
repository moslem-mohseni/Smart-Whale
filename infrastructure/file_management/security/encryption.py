import os
from cryptography.fernet import Fernet


class FileEncryption:
    """
    مدیریت رمزنگاری و رمزگشایی فایل‌ها
    """

    def __init__(self):
        self.secret_key = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())
        self.cipher = Fernet(self.secret_key)

    def encrypt(self, data: bytes) -> bytes:
        """رمزنگاری داده"""
        return self.cipher.encrypt(data)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """رمزگشایی داده"""
        return self.cipher.decrypt(encrypted_data)
