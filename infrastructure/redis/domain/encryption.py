from cryptography.fernet import Fernet
import base64
import pickle
from typing import Any, Optional


class EncryptedRedisAdapter:
    """
    مدیریت رمزنگاری داده‌های ذخیره‌شده در Redis برای افزایش امنیت
    """
    def __init__(self, redis_adapter, encryption_key: Optional[bytes] = None):
        self.redis_adapter = redis_adapter
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """رمزنگاری و ذخیره مقدار در Redis"""
        serialized = pickle.dumps(value)
        encrypted = self.fernet.encrypt(serialized)
        await self.redis_adapter.set(key, encrypted, ttl)

    async def get(self, key: str) -> Optional[Any]:
        """بازیابی و رمزگشایی مقدار از Redis"""
        encrypted = await self.redis_adapter.get(key)
        if encrypted is None:
            return None
        decrypted = self.fernet.decrypt(encrypted)
        return pickle.loads(decrypted)

    async def delete(self, key: str) -> bool:
        """حذف مقدار رمزنگاری‌شده از Redis"""
        return await self.redis_adapter.delete(key)

# نصب بسته مورد نیاز:
# pip install cryptography
