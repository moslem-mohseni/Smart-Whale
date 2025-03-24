import os
import jwt
from datetime import datetime, timedelta


class AccessControl:
    """
    مدیریت کنترل دسترسی فایل‌ها با استفاده از JWT
    """

    def __init__(self):
        self.secret_key = os.getenv("ACCESS_CONTROL_SECRET", "default_secret")
        self.token_expiry = int(os.getenv("ACCESS_TOKEN_EXPIRY", 3600))

    def generate_token(self, user_id: str, permissions: list) -> str:
        """تولید JWT برای کاربر با سطح دسترسی مشخص"""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": datetime.utcnow() + timedelta(seconds=self.token_expiry)
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> dict:
        """بررسی صحت JWT و بازگرداندن اطلاعات آن"""
        try:
            decoded = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return decoded
        except jwt.ExpiredSignatureError:
            return {"error": "Token has expired"}
        except jwt.InvalidTokenError:
            return {"error": "Invalid token"}
