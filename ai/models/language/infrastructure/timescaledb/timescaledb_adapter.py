import logging
from typing import List, Dict, Optional
from infrastructure.timescaledb.service.database_service import DatabaseService

class TimescaleDBAdapter:
    """
    این کلاس مدیریت ارتباط با TimescaleDB و عملیات سری‌زمانی را بر عهده دارد.
    این ماژول به‌صورت مستقیم از `DatabaseService` که در `infrastructure/timescaledb/` پیاده‌سازی شده است، استفاده می‌کند.
    """

    def __init__(self, database_service: DatabaseService):
        self.database_service = database_service
        logging.info("✅ TimescaleDBAdapter مقداردهی شد و ارتباط با DatabaseService برقرار شد.")

    async def insert_time_series_data(self, table_name: str, data: Dict[str, any]):
        """
        درج داده‌های سری‌زمانی در TimescaleDB.

        :param table_name: نام جدولی که داده در آن ذخیره می‌شود.
        :param data: داده‌ی سری‌زمانی شامل فیلدهای مختلف.
        """
        try:
            await self.database_service.store_time_series_data(table_name, **data)
            logging.info(f"✅ داده‌ی سری‌زمانی در جدول {table_name} ذخیره شد.")
        except Exception as e:
            logging.error(f"❌ خطا در درج داده‌های سری‌زمانی در TimescaleDB: {e}")

    async def get_time_series_data(self, table_name: str, start_time: str, end_time: str) -> Optional[List[Dict]]:
        """
        دریافت داده‌های سری‌زمانی در یک بازه زمانی.

        :param table_name: نام جدول.
        :param start_time: زمان شروع بازه.
        :param end_time: زمان پایان بازه.
        :return: لیستی از داده‌های سری‌زمانی.
        """
        try:
            data = await self.database_service.get_time_series_data(table_name, start_time, end_time)
            logging.info(f"📊 داده‌های سری‌زمانی از جدول {table_name} دریافت شد.")
            return data
        except Exception as e:
            logging.error(f"❌ خطا در دریافت داده‌های سری‌زمانی از TimescaleDB: {e}")
            return None

    async def delete_old_data(self, table_name: str):
        """
        حذف داده‌های قدیمی از جدول TimescaleDB.

        :param table_name: نام جدولی که داده‌های قدیمی از آن حذف می‌شوند.
        """
        try:
            await self.database_service.apply_retention_policy(table_name)
            logging.info(f"🗑 داده‌های قدیمی از جدول {table_name} حذف شد.")
        except Exception as e:
            logging.error(f"❌ خطا در حذف داده‌های قدیمی از TimescaleDB: {e}")
