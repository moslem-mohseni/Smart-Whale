import asyncio
import logging
from infrastructure.timescaledb.service.data_retention import DataRetention
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage
from infrastructure.timescaledb.config.connection_pool import ConnectionPool
from infrastructure.timescaledb.config.settings import TimescaleDBConfig

logger = logging.getLogger(__name__)


async def cleanup_old_data():
    """
    اجرای فرآیند حذف داده‌های قدیمی از پایگاه داده
    """
    logger.info("🗑️ شروع فرآیند حذف داده‌های قدیمی...")

    # مقداردهی اولیه ماژول TimescaleDB
    config = TimescaleDBConfig()
    connection_pool = ConnectionPool(config)
    await connection_pool.initialize()

    storage = TimescaleDBStorage(connection_pool)
    retention_service = DataRetention(storage)

    try:
        await retention_service.apply_retention_policy("time_series_data")
        logger.info("✅ داده‌های قدیمی با موفقیت حذف شدند.")
    except Exception as e:
        logger.error(f"❌ خطا در حذف داده‌های قدیمی: {e}")

    await connection_pool.close()

# اجرای اسکریپت
if __name__ == "__main__":
    asyncio.run(cleanup_old_data())
