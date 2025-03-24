import logging
from typing import List, Dict, Any, Optional
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

logger = logging.getLogger(__name__)


class AuditLog:
    """مدیریت ثبت و بررسی لاگ‌های امنیتی کاربران در TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
        """
        self.storage = storage

    async def create_audit_table(self):
        """
        ایجاد جدول ثبت لاگ‌های امنیتی در صورت عدم وجود
        """
        create_table_query = """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                details JSONB DEFAULT '{}'::JSONB,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """

        try:
            logger.info("🚀 ایجاد جدول `audit_logs` در پایگاه داده...")
            await self.storage.execute_query(create_table_query)
            logger.info("✅ جدول `audit_logs` با موفقیت ایجاد شد.")
        except Exception as e:
            logger.error(f"❌ خطا در ایجاد جدول لاگ‌های امنیتی: {e}")
            raise

    async def log_action(self, username: str, action: str, details: Optional[Dict[str, Any]] = None):
        """
        ثبت یک عملیات کاربر در جدول لاگ‌ها

        Args:
            username (str): نام کاربری
            action (str): نوع عملیات (مثلاً "LOGIN", "UPDATE_RECORD", "DELETE_USER")
            details (Optional[Dict[str, Any]]): اطلاعات اضافی درباره عملیات
        """
        log_query = """
            INSERT INTO audit_logs (username, action, details)
            VALUES ($1, $2, $3);
        """

        try:
            logger.info(f"📝 ثبت لاگ امنیتی برای کاربر `{username}`: {action}")
            await self.storage.execute_query(log_query, [username, action, details or {}])
            logger.info("✅ لاگ عملیات ثبت شد.")
        except Exception as e:
            logger.error(f"❌ خطا در ثبت لاگ امنیتی: {e}")
            raise

    async def get_logs(self, username: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        بازیابی لاگ‌های امنیتی بر اساس نام کاربر و محدوده زمانی

        Args:
            username (Optional[str]): نام کاربری (در صورت خالی بودن، همه کاربران بررسی می‌شوند)
            start_time (Optional[str]): تاریخ شروع جستجو (فرمت: 'YYYY-MM-DD HH:MI:SS')
            end_time (Optional[str]): تاریخ پایان جستجو

        Returns:
            List[Dict[str, Any]]: لیستی از لاگ‌های امنیتی
        """
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []

        if username:
            query += " AND username = $1"
            params.append(username)
        if start_time and end_time:
            query += " AND timestamp BETWEEN $2 AND $3"
            params.extend([start_time, end_time])

        try:
            logger.info(f"🔍 بازیابی لاگ‌های امنیتی برای کاربر `{username or 'همه'}` در بازه `{start_time} - {end_time}`...")
            logs = await self.storage.execute_query(query, params)
            logger.info("✅ لاگ‌های امنیتی بازیابی شدند.")
            return logs
        except Exception as e:
            logger.error(f"❌ خطا در بازیابی لاگ‌های امنیتی: {e}")
            return []

    async def delete_old_logs(self, retention_period: str = "90 days"):
        """
        حذف لاگ‌های قدیمی‌تر از یک محدوده مشخص

        Args:
            retention_period (str): مدت زمان نگهداری لاگ‌ها قبل از حذف (مثلاً '90 days')
        """
        delete_query = f"""
            DELETE FROM audit_logs WHERE timestamp < NOW() - INTERVAL '{retention_period}';
        """

        try:
            logger.info(f"🗑️ حذف لاگ‌های قدیمی‌تر از `{retention_period}`...")
            await self.storage.execute_query(delete_query)
            logger.info("✅ لاگ‌های قدیمی حذف شدند.")
        except Exception as e:
            logger.error(f"❌ خطا در حذف لاگ‌های قدیمی: {e}")
            raise
