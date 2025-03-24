import jwt
import logging
import datetime
from typing import Optional
from infrastructure.redis.service.cache_service import CacheService

class AuthManager:
    SECRET_KEY = "supersecretkey"  # این مقدار در محیط واقعی باید در ENV ذخیره شود
    ALGORITHM = "HS256"
    TOKEN_EXPIRATION_MINUTES = 60

    def __init__(self, redis_client: CacheService):
        """
        مدیریت احراز هویت کاربران با استفاده از JWT
        :param redis_client: سرویس Redis برای ذخیره و مدیریت توکن‌های منقضی‌شده
        """
        self.redis = redis_client
        self.logger = logging.getLogger("AuthManager")

    def generate_token(self, user_id: str) -> str:
        """ تولید یک توکن JWT برای کاربر """
        expiration_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=self.TOKEN_EXPIRATION_MINUTES)
        payload = {"user_id": user_id, "exp": expiration_time}
        token = jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return token

    def verify_token(self, token: str) -> Optional[dict]:
        """ بررسی اعتبار توکن JWT """
        try:
            # بررسی اینکه آیا توکن در لیست سیاه (Blacklist) قرار دارد یا نه
            if self.redis.get(f"revoked_token:{token}"):
                self.logger.warning("❌ تلاش برای استفاده از توکن لغو شده!")
                return None

            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("⚠️ توکن منقضی شده است.")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("❌ توکن نامعتبر است.")
            return None

    def revoke_token(self, token: str):
        """ اضافه کردن توکن به لیست سیاه (Blacklist) برای جلوگیری از استفاده مجدد """
        self.redis.set(f"revoked_token:{token}", "revoked", ex=self.TOKEN_EXPIRATION_MINUTES * 60)
        self.logger.info("✅ توکن با موفقیت لغو شد.")
