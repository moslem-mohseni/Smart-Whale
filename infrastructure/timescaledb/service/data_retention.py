import logging
import os
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

logger = logging.getLogger(__name__)


class DataRetention:
    """مدیریت سیاست‌های نگهداری داده در TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
        """
        self.storage = storage
        self.retention_period = os.getenv("DATA_RETENTION_PERIOD", "30 days")  # مقدار پیش‌فرض ۳۰ روز

    async def apply_retention_policy(self, table_name: str):
        """
        اعمال سیاست حذف خودکار داده‌های قدیمی

        Args:
            table_name (str): نام جدول داده‌های سری‌زمانی
        """
        query = f"""
            SELECT add_retention_policy('{table_name}', INTERVAL '{self.retention_period}');
        """

        try:
            logger.info(f"🗑️ تنظیم سیاست حذف داده‌های قدیمی برای `{table_name}`...")
            await self.storage.execute_query(query)
            logger.info("✅ سیاست نگهداری داده‌ها اعمال شد.")
        except Exception as e:
            logger.error(f"❌ خطا در تنظیم سیاست نگهداری داده‌ها: {e}")
            raise
