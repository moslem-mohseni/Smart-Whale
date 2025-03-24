import logging
import os
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

logger = logging.getLogger(__name__)


class DataCompressor:
    """مدیریت فشرده‌سازی داده‌های سری‌زمانی در TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
        """
        self.storage = storage
        self.segment_by = os.getenv("COMPRESSION_SEGMENT_BY", "id")
        self.order_by = os.getenv("COMPRESSION_ORDER_BY", "timestamp DESC")
        self.compression_threshold = os.getenv("COMPRESSION_THRESHOLD", "7 days")

    async def enable_compression(self, table_name: str):
        """
        فعال‌سازی قابلیت فشرده‌سازی برای یک جدول

        Args:
            table_name (str): نام جدول
        """
        enable_compression_query = f"""
            ALTER TABLE {table_name} SET (
                timescaledb.compress = TRUE,
                timescaledb.compress_segmentby = '{self.segment_by}',
                timescaledb.compress_orderby = '{self.order_by}'
            );
        """

        try:
            logger.info(f"🚀 فعال‌سازی فشرده‌سازی برای جدول `{table_name}`...")
            await self.storage.execute_query(enable_compression_query)
            logger.info("✅ قابلیت فشرده‌سازی فعال شد.")
        except Exception as e:
            logger.error(f"❌ خطا در فعال‌سازی فشرده‌سازی: {e}")
            raise

    async def compress_old_chunks(self, table_name: str):
        """
        فشرده‌سازی داده‌های قدیمی‌تر از `COMPRESSION_THRESHOLD`

        Args:
            table_name (str): نام جدول
        """
        compress_chunks_query = f"""
            SELECT compress_chunk(chunk)
            FROM show_chunks('{table_name}', older_than => INTERVAL '{self.compression_threshold}');
        """

        try:
            logger.info(f"📦 فشرده‌سازی داده‌های قدیمی در `{table_name}`...")
            await self.storage.execute_query(compress_chunks_query)
            logger.info("✅ داده‌های قدیمی فشرده شدند.")
        except Exception as e:
            logger.error(f"❌ خطا در فشرده‌سازی داده‌ها: {e}")
            raise
