import logging
from typing import Optional, List, Dict
from infrastructure.clickhouse.adapters.clickhouse_adapter import ClickHouseAdapter
from infrastructure.clickhouse.optimization.cache_manager import CacheManager
from infrastructure.clickhouse.optimization.query_optimizer import QueryOptimizer

class ClickHouseDB:
    """
    این کلاس مدیریت ارتباط با ClickHouse برای اجرای کوئری‌های تحلیلی و ذخیره داده‌های پردازشی زبان را بر عهده دارد.
    """

    def __init__(self, clickhouse_adapter: ClickHouseAdapter, cache_manager: CacheManager, query_optimizer: QueryOptimizer):
        self.clickhouse_adapter = clickhouse_adapter
        self.cache_manager = cache_manager
        self.query_optimizer = query_optimizer
        logging.info("✅ ClickHouseDB مقداردهی شد و ارتباط با ClickHouse برقرار شد.")

    async def execute_query(self, query: str) -> Optional[List[Dict]]:
        """
        اجرای کوئری تحلیلی در ClickHouse.

        1️⃣ ابتدا بررسی می‌شود که نتیجه‌ی کوئری در `CacheManager` موجود است یا نه.
        2️⃣ در صورت نبود، کوئری با `QueryOptimizer` بهینه‌سازی می‌شود.
        3️⃣ سپس کوئری به ClickHouse ارسال شده و نتیجه ذخیره می‌شود.

        :param query: متن کوئری SQL
        :return: نتیجه‌ی کوئری در قالب لیستی از دیکشنری‌ها
        """
        try:
            cached_result = await self.cache_manager.get_cached_result(query)
            if cached_result:
                logging.info(f"📥 نتیجه‌ی کوئری از کش دریافت شد.")
                return cached_result

            optimized_query = await self.query_optimizer.optimize(query)
            result = await self.clickhouse_adapter.execute(optimized_query)

            if result:
                await self.cache_manager.cache_result(query, result, ttl=600)  # ذخیره در کش به مدت ۱۰ دقیقه
                logging.info(f"✅ کوئری اجرا شد و نتیجه در کش ذخیره شد.")
            return result

        except Exception as e:
            logging.error(f"❌ خطا در اجرای کوئری در ClickHouse: {e}")
            return None

    async def insert_data(self, table_name: str, data: List[Dict]):
        """
        درج داده‌ها در جدول ClickHouse.

        :param table_name: نام جدول
        :param data: داده‌هایی که باید درج شوند
        """
        try:
            await self.clickhouse_adapter.insert_data(table_name, data)
            logging.info(f"✅ داده‌ها در جدول `{table_name}` ذخیره شدند.")
        except Exception as e:
            logging.error(f"❌ خطا در درج داده در ClickHouse: {e}")

    async def delete_old_data(self, table_name: str, retention_period: int):
        """
        حذف داده‌های قدیمی از جدول ClickHouse.

        :param table_name: نام جدول
        :param retention_period: مدت نگهداری داده‌ها بر حسب روز
        """
        try:
            query = f"DELETE FROM {table_name} WHERE timestamp < now() - INTERVAL {retention_period} DAY"
            await self.execute_query(query)
            logging.info(f"🗑 داده‌های قدیمی‌تر از {retention_period} روز از `{table_name}` حذف شدند.")
        except Exception as e:
            logging.error(f"❌ خطا در حذف داده‌های قدیمی از ClickHouse: {e}")
