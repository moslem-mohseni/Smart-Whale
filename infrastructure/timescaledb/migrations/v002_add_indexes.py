import logging
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage
from infrastructure.timescaledb.config.connection_pool import ConnectionPool
from infrastructure.timescaledb.config.settings import TimescaleDBConfig

logger = logging.getLogger(__name__)


class IndexMigration:
    """مدیریت ایجاد ایندکس‌های بهینه در TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی TimescaleDB
        """
        self.storage = storage

    async def apply_migration(self):
        """اجرای مهاجرت ایندکس‌ها"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON time_series_data (timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_metadata ON time_series_data USING GIN (metadata);"
        ]

        try:
            logger.info("🚀 ایجاد ایندکس‌های بهینه برای `time_series_data`...")
            for query in index_queries:
                await self.storage.execute_query(query)
            logger.info("✅ ایندکس‌ها با موفقیت ایجاد شدند.")
        except Exception as e:
            logger.error(f"❌ خطا در ایجاد ایندکس‌ها: {e}")
            raise


# اجرای اسکریپت به صورت مستقل
if __name__ == "__main__":
    import asyncio

    async def run_migration():
        config = TimescaleDBConfig()
        connection_pool = ConnectionPool(config)
        await connection_pool.initialize()

        storage = TimescaleDBStorage(connection_pool)
        migration = IndexMigration(storage)
        await migration.apply_migration()

        await connection_pool.close()

    asyncio.run(run_migration())
