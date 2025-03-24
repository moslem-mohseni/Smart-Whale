# infrastructure/clickhouse/security/access_control.py
import jwt
import os
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from ..config.config import config

logger = logging.getLogger(__name__)


class AccessControl:
    """
    مدیریت کنترل دسترسی کاربران در ClickHouse

    این کلاس وظیفه مدیریت توکن‌های JWT برای احراز هویت و تأیید دسترسی‌ها را بر عهده دارد.
    """

    def __init__(self, secret_key: Optional[str] = None, token_expiry: Optional[int] = None):
        """
        مقداردهی اولیه سیستم کنترل دسترسی

        Args:
            secret_key (str, optional): کلید رمزنگاری توکن‌های JWT. اگر مشخص نشده باشد، از تنظیمات مرکزی استفاده می‌شود.
            token_expiry (int, optional): مدت اعتبار توکن (ثانیه). اگر مشخص نشده باشد، از تنظیمات مرکزی استفاده می‌شود.
        """
        # استفاده از مقادیر ارسالی یا مقادیر تنظیمات مرکزی
        self.secret_key = secret_key or config.access_control_secret
        self.token_expiry = token_expiry or config.access_token_expiry

        # بررسی اعتبار کلید محرمانه
        if not self.secret_key or len(self.secret_key) < 32:
            error_msg = "Secret key is missing or too short (min 32 chars). JWT operations are not secure!"
            logger.error(error_msg)

            # در محیط تولید، کلید پیش‌فرض نباید استفاده شود
            if os.getenv('APP_ENV') == 'production':
                raise ValueError(error_msg)
            else:
                # در محیط توسعه، یک کلید تصادفی تولید می‌کنیم
                logger.warning("Generating a random secret key for development. DO NOT USE IN PRODUCTION.")
                self.secret_key = secrets.token_hex(32)  # 64 کاراکتر هگزادسیمال

        logger.info(f"Access control initialized with token expiry: {self.token_expiry} seconds")

    def generate_token(self, username: str, role: str, custom_claims: Optional[Dict[str, Any]] = None) -> str:
        """
        تولید توکن JWT برای احراز هویت

        Args:
            username (str): نام کاربری
            role (str): نقش کاربر
            custom_claims (Dict[str, Any], optional): ادعاهای سفارشی اضافی

        Returns:
            str: توکن JWT
        """
        # زمان انقضا
        exp_time = datetime.utcnow() + timedelta(seconds=self.token_expiry)

        # ساخت پیلود پایه
        payload = {
            "sub": username,  # موضوع توکن
            "username": username,
            "role": role,
            "exp": exp_time,  # زمان انقضا
            "iat": datetime.utcnow(),  # زمان صدور
            "iss": "clickhouse-service"  # صادرکننده
        }

        # افزودن ادعاهای سفارشی
        if custom_claims:
            # اطمینان از عدم بازنویسی فیلدهای اصلی
            reserved_keys = ["sub", "exp", "iat", "iss", "nbf", "aud", "jti"]
            for key in reserved_keys:
                if key in custom_claims:
                    logger.warning(f"Reserved JWT claim '{key}' in custom_claims will be ignored")
                    del custom_claims[key]

            payload.update(custom_claims)

        # رمزنگاری و تولید توکن
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")

        logger.debug(f"Generated token for user '{username}' with role '{role}', expires at {exp_time}")
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        بررسی صحت توکن JWT

        Args:
            token (str): توکن JWT

        Returns:
            Dict[str, Any] | None: اطلاعات توکن در صورت معتبر بودن، یا None در صورت نامعتبر بودن
        """
        if not token:
            logger.warning("Empty token provided for verification")
            return None

        try:
            # رمزگشایی و تأیید توکن
            decoded_token = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"],
                options={"verify_signature": True, "verify_exp": True}
            )
            logger.debug(f"Token verified successfully for user '{decoded_token.get('username')}'")
            return decoded_token

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None

        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error during token verification: {str(e)}")
            return None

    def get_permissions(self, token: str) -> Dict[str, Any]:
        """
        استخراج مجوزهای کاربر از توکن

        Args:
            token (str): توکن JWT

        Returns:
            Dict[str, Any]: دیکشنری شامل مجوزهای کاربر یا دیکشنری خالی در صورت نامعتبر بودن توکن
        """
        decoded = self.verify_token(token)
        if not decoded:
            return {}

        # استخراج نقش و سایر اطلاعات مربوط به مجوز
        role = decoded.get('role', '')
        permissions = decoded.get('permissions', {})

        # تبدیل نقش به مجوزهای مربوطه
        if role == 'admin':
            # مدیران دسترسی کامل دارند
            permissions['full_access'] = True
        elif role == 'reader':
            # کاربران فقط-خواندنی
            permissions['read'] = True
        elif role == 'writer':
            # کاربران با دسترسی نوشتن
            permissions['read'] = True
            permissions['write'] = True

        return permissions

    def refresh_token(self, token: str) -> Optional[str]:
        """
        تجدید توکن قبل از انقضا

        Args:
            token (str): توکن فعلی

        Returns:
            str | None: توکن جدید یا None در صورت نامعتبر بودن توکن فعلی
        """
        decoded = self.verify_token(token)
        if not decoded:
            return None

        # تجدید توکن با استفاده از اطلاعات موجود
        username = decoded.get('username', '')
        role = decoded.get('role', '')

        # حذف فیلدهای مربوط به زمان
        custom_claims = {k: v for k, v in decoded.items()
                         if k not in ['exp', 'iat', 'sub', 'username', 'role', 'iss']}

        # تولید توکن جدید
        return self.generate_token(username, role, custom_claims)