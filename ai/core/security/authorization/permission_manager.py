import logging
from typing import List, Optional
from infrastructure.redis.service.cache_service import CacheService

class PermissionManager:
    def __init__(self, redis_client: CacheService):
        """
        مدیریت مجوزهای دسترسی کاربران
        :param redis_client: سرویس Redis برای ذخیره مجوزهای کاربران
        """
        self.redis = redis_client
        self.logger = logging.getLogger("PermissionManager")

    def set_permissions(self, user_id: str, permissions: List[str]):
        """ تنظیم مجوزهای یک کاربر در Redis """
        self.redis.set(f"user_permissions:{user_id}", ",".join(permissions))
        self.logger.info(f"✅ مجوزهای کاربر {user_id} تنظیم شد: {permissions}")

    def get_permissions(self, user_id: str) -> Optional[List[str]]:
        """ دریافت مجوزهای یک کاربر از Redis """
        permissions = self.redis.get(f"user_permissions:{user_id}")
        return permissions.split(",") if permissions else None

    def has_permission(self, user_id: str, permission: str) -> bool:
        """ بررسی اینکه آیا یک کاربر مجوز خاصی دارد یا نه """
        permissions = self.get_permissions(user_id)
        if permissions and permission in permissions:
            return True
        self.logger.warning(f"⛔️ کاربر {user_id} دسترسی به {permission} را ندارد.")
        return False
