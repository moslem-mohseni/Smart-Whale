import jwt
import logging
import datetime
from typing import Optional
from infrastructure.redis.service.cache_service import CacheService

class TokenManager:
    SECRET_KEY = "supersecretkey"  # این مقدار باید در متغیرهای محیطی ذخیره شود
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRATION_MINUTES = 60
    REFRESH_TOKEN_EXPIRATION_DAYS = 7

    def __init__(self, redis_client: CacheService):
        """
        مدیریت توکن‌های JWT برای احراز هویت کاربران
        :param redis_client: سرویس Redis برای مدیریت توکن‌های لغو شده
        """
        self.redis = redis_client
        self.logger = logging.getLogger("TokenManager")

    def generate_access_token(self, user_id: str) -> str:
        """ تولید توکن دسترسی (Access Token) """
        expiration_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=self.ACCESS_TOKEN_EXPIRATION_MINUTES)
        payload = {"user_id": user_id, "exp": expiration_time, "type": "access"}
        token = jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return token

    def generate_refresh_token(self, user_id: str) -> str:
        """ تولید توکن تمدید اعتبار (Refresh Token) """
        expiration_time = datetime.datetime.utcnow() + datetime.timedelta(days=self.REFRESH_TOKEN_EXPIRATION_DAYS)
        payload = {"user_id": user_id, "exp": expiration_time, "type": "refresh"}
        token = jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return token

    def verify_token(self, token: str) -> Optional[dict]:
        """ بررسی اعتبار توکن JWT """
        try:
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

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """ تمدید توکن دسترسی با استفاده از توکن تمدید اعتبار """
        payload = self.verify_token(refresh_token)
        if payload and payload.get("type") == "refresh":
            return self.generate_access_token(payload["user_id"])
        self.logger.warning("❌ توکن تمدید اعتبار نامعتبر است.")
        return None

    def revoke_token(self, token: str):
        """ اضافه کردن توکن به لیست سیاه (Blacklist) برای جلوگیری از استفاده مجدد """
        self.redis.set(f"revoked_token:{token}", "revoked", ex=self.ACCESS_TOKEN_EXPIRATION_MINUTES * 60)
        self.logger.info("✅ توکن با موفقیت لغو شد.")
