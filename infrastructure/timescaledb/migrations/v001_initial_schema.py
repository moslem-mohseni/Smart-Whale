import logging
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage
from infrastructure.timescaledb.config.connection_pool import ConnectionPool
from infrastructure.timescaledb.config.settings import TimescaleDBConfig

logger = logging.getLogger(__name__)


class InitialSchemaMigration:
    """مدیریت مهاجرت اولیه پایگاه داده"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی TimescaleDB
        """
        self.storage = storage

    async def apply_migration(self):
        """اجرای مهاجرت اولیه"""
        create_table_query = """
            CREATE TABLE IF NOT EXISTS time_series_data (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                value DOUBLE PRECISION NOT NULL,
                metadata JSONB DEFAULT '{}'::JSONB
            );
        """

        convert_to_hypertable_query = """
            SELECT create_hypertable('time_series_data', 'timestamp', if_not_exists => TRUE);
        """

        try:
            logger.info("🚀 ایجاد جدول اولیه `time_series_data`...")
            await self.storage.execute_query(create_table_query)
            logger.info("✅ جدول `time_series_data` با موفقیت ایجاد شد.")

            logger.info("🚀 تبدیل جدول به Hypertable...")
            await self.storage.execute_query(convert_to_hypertable_query)
            logger.info("✅ جدول `time_series_data` به Hypertable تبدیل شد.")
        except Exception as e:
            logger.error(f"❌ خطا در مهاجرت اولیه پایگاه داده: {e}")
            raise


# اجرای اسکریپت به صورت مستقل
if __name__ == "__main__":
    import asyncio
    async def run_migration():
        config = TimescaleDBConfig()
        connection_pool = ConnectionPool(config)
        await connection_pool.initialize()

        storage = TimescaleDBStorage(connection_pool)
        migration = InitialSchemaMigration(storage)
        await migration.apply_migration()

        await connection_pool.close()

    asyncio.run(run_migration())
