import logging
from typing import List
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

logger = logging.getLogger(__name__)


class AccessControl:
    """مدیریت کنترل دسترسی و نقش‌های کاربری در TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
        """
        self.storage = storage

    async def create_role(self, role_name: str, privileges: List[str]):
        """
        ایجاد یک نقش جدید با مجوزهای مشخص

        Args:
            role_name (str): نام نقش
            privileges (List[str]): لیست مجوزها (SELECT, INSERT, UPDATE, DELETE)
        """
        privilege_str = ", ".join(privileges)
        create_role_query = f"CREATE ROLE {role_name};"
        grant_privileges_query = f"GRANT {privilege_str} ON ALL TABLES IN SCHEMA public TO {role_name};"

        try:
            logger.info(f"🚀 ایجاد نقش `{role_name}` با مجوزهای: {privileges}...")
            await self.storage.execute_query(create_role_query)
            await self.storage.execute_query(grant_privileges_query)
            logger.info("✅ نقش با موفقیت ایجاد شد.")
        except Exception as e:
            logger.error(f"❌ خطا در ایجاد نقش: {e}")
            raise

    async def assign_role_to_user(self, username: str, role_name: str):
        """
        اختصاص یک نقش به کاربر

        Args:
            username (str): نام کاربری
            role_name (str): نام نقش
        """
        assign_query = f"GRANT {role_name} TO {username};"

        try:
            logger.info(f"👤 اختصاص نقش `{role_name}` به کاربر `{username}`...")
            await self.storage.execute_query(assign_query)
            logger.info("✅ نقش با موفقیت به کاربر اختصاص داده شد.")
        except Exception as e:
            logger.error(f"❌ خطا در اختصاص نقش: {e}")
            raise

    async def revoke_role_from_user(self, username: str, role_name: str):
        """
        حذف نقش از کاربر

        Args:
            username (str): نام کاربری
            role_name (str): نام نقش
        """
        revoke_query = f"REVOKE {role_name} FROM {username};"

        try:
            logger.info(f"❌ حذف نقش `{role_name}` از کاربر `{username}`...")
            await self.storage.execute_query(revoke_query)
            logger.info("✅ نقش با موفقیت از کاربر حذف شد.")
        except Exception as e:
            logger.error(f"❌ خطا در حذف نقش از کاربر: {e}")
            raise

    async def check_user_privileges(self, username: str) -> List[str]:
        """
        بررسی مجوزهای کاربر

        Args:
            username (str): نام کاربری

        Returns:
            List[str]: لیستی از مجوزهای کاربر
        """
        check_query = f"""
            SELECT grantee, privilege_type 
            FROM information_schema.role_table_grants 
            WHERE grantee = '{username}';
        """

        try:
            result = await self.storage.execute_query(check_query)
            privileges = [row["privilege_type"] for row in result]
            logger.info(f"🔍 مجوزهای کاربر `{username}`: {privileges}")
            return privileges
        except Exception as e:
            logger.error(f"❌ خطا در بررسی مجوزهای کاربر: {e}")
            return []
