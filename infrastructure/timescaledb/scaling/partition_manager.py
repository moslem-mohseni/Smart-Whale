import logging
import os
from typing import Optional, List
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

logger = logging.getLogger(__name__)


class PartitionManager:
    """مدیریت Sharding و Partitioning در TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
        """
        self.storage = storage
        self.partition_interval = os.getenv("DEFAULT_PARTITION_INTERVAL", "7 days")
        self.retention_policy = os.getenv("DATA_RETENTION_POLICY", "30 days")  # مدت نگهداری داده‌ها قبل از حذف

    async def create_partition_policy(self, table_name: str):
        """
        تنظیم پارتیشن‌بندی خودکار برای یک جدول

        Args:
            table_name (str): نام جدول
        """
        partition_query = f"""
            SELECT add_retention_policy('{table_name}', INTERVAL '{self.retention_policy}');
        """

        try:
            logger.info(f"🚀 تنظیم پارتیشن‌بندی خودکار برای `{table_name}`...")
            await self.storage.execute_query(partition_query)
            logger.info("✅ پارتیشن‌بندی خودکار فعال شد.")
        except Exception as e:
            logger.error(f"❌ خطا در تنظیم پارتیشن‌بندی: {e}")
            raise

    async def check_partitions(self, table_name: str) -> Optional[List[str]]:
        """
        بررسی پارتیشن‌های موجود در یک جدول

        Args:
            table_name (str): نام جدول

        Returns:
            Optional[List[str]]: لیستی از پارتیشن‌های موجود یا None در صورت عدم وجود
        """
        query = f"""
            SELECT show_chunks('{table_name}');
        """

        try:
            result = await self.storage.execute_query(query)
            if result:
                partitions = [row["show_chunks"] for row in result]
                logger.info(f"📡 پارتیشن‌های فعال برای `{table_name}`: {partitions}")
                return partitions
            else:
                logger.warning(f"⚠️ هیچ پارتیشنی برای `{table_name}` یافت نشد.")
                return None
        except Exception as e:
            logger.error(f"❌ خطا در بررسی پارتیشن‌ها: {e}")
            return None

    async def drop_old_partitions(self, table_name: str):
        """
        حذف پارتیشن‌های قدیمی برای آزادسازی فضای ذخیره‌سازی

        Args:
            table_name (str): نام جدول
        """
        drop_query = f"""
            SELECT drop_chunks('{table_name}', INTERVAL '{self.retention_policy}');
        """

        try:
            logger.info(f"🗑️ حذف پارتیشن‌های قدیمی در `{table_name}`...")
            await self.storage.execute_query(drop_query)
            logger.info("✅ پارتیشن‌های قدیمی حذف شدند.")
        except Exception as e:
            logger.error(f"❌ خطا در حذف پارتیشن‌های قدیمی: {e}")
            raise
