import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
from infrastructure.timescaledb.service.database_service import DatabaseService

class MetricsHandler:
    """
    این کلاس وظیفه‌ی مدیریت متریک‌های ذخیره‌سازی و بازیابی داده‌های سری‌زمانی در TimescaleDB را دارد.
    """

    def __init__(self, database_service: DatabaseService):
        self.database_service = database_service
        logging.info("✅ MetricsHandler مقداردهی شد و ارتباط با DatabaseService برقرار شد.")

    async def get_storage_metrics(self, table_name: str) -> Optional[Dict]:
        """
        دریافت متریک‌های مربوط به حجم ذخیره‌سازی و تعداد رکوردها.

        :param table_name: نام جدول TimescaleDB.
        :return: دیکشنری شامل متریک‌های ذخیره‌سازی.
        """
        try:
            query = f"""
                SELECT pg_size_pretty(pg_total_relation_size('{table_name}')) AS table_size,
                       COUNT(*) AS total_records
                FROM {table_name}
            """
            result = await self.database_service.execute_query(query)
            if result:
                logging.info(f"📊 متریک‌های ذخیره‌سازی از جدول {table_name} دریافت شد.")
                return result[0]  # اولین سطر از نتیجه‌ی کوئری بازگردانده می‌شود.
            else:
                logging.warning(f"⚠️ متریک‌های ذخیره‌سازی برای جدول {table_name} یافت نشد.")
                return None
        except Exception as e:
            logging.error(f"❌ خطا در دریافت متریک‌های ذخیره‌سازی از TimescaleDB: {e}")
            return None

    async def get_query_performance_metrics(self, start_time: datetime, end_time: datetime) -> Optional[Dict]:
        """
        دریافت متریک‌های عملکردی کوئری‌های اجراشده در یک بازه زمانی.

        :param start_time: زمان شروع بازه.
        :param end_time: زمان پایان بازه.
        :return: دیکشنری شامل متریک‌های کوئری.
        """
        try:
            query = f"""
                SELECT COUNT(*) AS total_queries,
                       AVG(execution_time) AS avg_execution_time,
                       MAX(execution_time) AS max_execution_time
                FROM query_log
                WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'
            """
            result = await self.database_service.execute_query(query)
            if result:
                logging.info(f"📊 متریک‌های عملکرد کوئری‌ها دریافت شد.")
                return result[0]
            else:
                logging.warning(f"⚠️ متریک‌های عملکردی کوئری‌ها در بازه‌ی زمانی مشخص‌شده یافت نشد.")
                return None
        except Exception as e:
            logging.error(f"❌ خطا در دریافت متریک‌های عملکردی کوئری‌ها: {e}")
            return None

    async def get_data_retention_status(self, table_name: str) -> Optional[Dict]:
        """
        بررسی وضعیت نگهداری داده‌ها و تعیین حجم داده‌های قدیمی.

        :param table_name: نام جدول TimescaleDB.
        :return: دیکشنری شامل اطلاعات داده‌های قدیمی.
        """
        try:
            query = f"""
                SELECT COUNT(*) AS old_records
                FROM {table_name}
                WHERE timestamp < NOW() - INTERVAL '30 days'
            """
            result = await self.database_service.execute_query(query)
            if result:
                logging.info(f"📊 وضعیت نگهداری داده‌ها برای جدول {table_name} بررسی شد.")
                return {"old_records": result[0]["old_records"]}
            else:
                logging.warning(f"⚠️ اطلاعاتی در مورد داده‌های قدیمی جدول {table_name} یافت نشد.")
                return None
        except Exception as e:
            logging.error(f"❌ خطا در بررسی وضعیت نگهداری داده‌ها: {e}")
            return None
