import logging
from cryptography.fernet import Fernet
from infrastructure.redis.service.cache_service import CacheService

class KeyManager:
    def __init__(self, redis_client: CacheService, key_name="encryption_key"):
        """
        مدیریت کلیدهای رمزنگاری
        :param redis_client: سرویس Redis برای ذخیره‌سازی کلیدهای رمزنگاری
        :param key_name: نام کلید ذخیره‌شده در Redis
        """
        self.redis = redis_client
        self.key_name = key_name
        self.logger = logging.getLogger("KeyManager")

    def generate_key(self) -> str:
        """ ایجاد کلید رمزنگاری جدید """
        key = Fernet.generate_key()
        self.redis.set(self.key_name, key.decode())
        self.logger.info(f"✅ کلید جدید رمزنگاری ایجاد و در Redis ذخیره شد: {self.key_name}")
        return key.decode()

    def get_key(self) -> str:
        """ بازیابی کلید رمزنگاری از Redis """
        key = self.redis.get(self.key_name)
        if key:
            return key
        self.logger.warning("⚠️ کلید رمزنگاری یافت نشد، کلید جدید تولید خواهد شد.")
        return self.generate_key()

    def rotate_key(self):
        """ چرخش (Rotation) کلید رمزنگاری برای امنیت بیشتر """
        new_key = self.generate_key()
        self.logger.info("🔄 کلید رمزنگاری با موفقیت چرخش یافت.")
        return new_key
