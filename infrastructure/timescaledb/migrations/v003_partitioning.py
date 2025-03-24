import logging
import os
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage
from infrastructure.timescaledb.config.connection_pool import ConnectionPool
from infrastructure.timescaledb.config.settings import TimescaleDBConfig

logger = logging.getLogger(__name__)


class PartitioningMigration:
    """مدیریت پیکربندی Sharding و Partitioning در TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی TimescaleDB
        """
        self.storage = storage
        self.partition_interval = os.getenv("DEFAULT_PARTITION_INTERVAL", "7 days")  # مقدار پیش‌فرض هفتگی

    async def apply_migration(self):
        """اجرای پیکربندی Partitioning"""
        partition_query = f"""
            ALTER TABLE time_series_data SET (
                timescaledb.chunk_time_interval = INTERVAL '{self.partition_interval}'
            );
        """

        try:
            logger.info(f"🚀 تنظیم Partitioning روی `{self.partition_interval}` برای `time_series_data`...")
            await self.storage.execute_query(partition_query)
            logger.info("✅ پارتیشن‌بندی با موفقیت اعمال شد.")
        except Exception as e:
            logger.error(f"❌ خطا در پارتیشن‌بندی: {e}")
            raise


# اجرای اسکریپت به صورت مستقل
if __name__ == "__main__":
    import asyncio

    async def run_migration():
        config = TimescaleDBConfig()
        connection_pool = ConnectionPool(config)
        await connection_pool.initialize()

        storage = TimescaleDBStorage(connection_pool)
        migration = PartitioningMigration(storage)
        await migration.apply_migration()

        await connection_pool.close()

    asyncio.run(run_migration())
