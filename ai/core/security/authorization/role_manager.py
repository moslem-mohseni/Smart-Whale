import logging
from typing import List, Optional
from infrastructure.redis.service.cache_service import CacheService


class RoleManager:
    def __init__(self, redis_client: CacheService):
        """
        مدیریت نقش‌های کاربران
        :param redis_client: سرویس Redis برای ذخیره نقش‌های کاربران
        """
        self.redis = redis_client
        self.logger = logging.getLogger("RoleManager")

    def set_roles(self, user_id: str, roles: List[str]):
        """ تنظیم نقش‌های یک کاربر در Redis """
        self.redis.set(f"user_roles:{user_id}", ",".join(roles))
        self.logger.info(f"✅ نقش‌های کاربر {user_id} تنظیم شد: {roles}")

    def get_roles(self, user_id: str) -> Optional[List[str]]:
        """ دریافت نقش‌های یک کاربر از Redis """
        roles = self.redis.get(f"user_roles:{user_id}")
        return roles.split(",") if roles else None

    def has_role(self, user_id: str, role: str) -> bool:
        """ بررسی اینکه آیا یک کاربر دارای یک نقش خاص است یا نه """
        roles = self.get_roles(user_id)
        if roles and role in roles:
            return True
        self.logger.warning(f"⛔️ کاربر {user_id} دارای نقش {role} نیست.")
        return False

    def add_role(self, user_id: str, role: str):
        """ افزودن یک نقش جدید به کاربر """
        roles = self.get_roles(user_id) or []
        if role not in roles:
            roles.append(role)
            self.set_roles(user_id, roles)
            self.logger.info(f"✅ نقش {role} به کاربر {user_id} اضافه شد.")

    def remove_role(self, user_id: str, role: str):
        """ حذف یک نقش از کاربر """
        roles = self.get_roles(user_id)
        if roles and role in roles:
            roles.remove(role)
            self.set_roles(user_id, roles)
            self.logger.info(f"✅ نقش {role} از کاربر {user_id} حذف شد.")
