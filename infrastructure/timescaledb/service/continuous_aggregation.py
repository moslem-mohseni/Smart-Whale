import logging
from typing import List
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

logger = logging.getLogger(__name__)


class ContinuousAggregation:
    """مدیریت تجمیع داده‌های سری‌زمانی در TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
        """
        self.storage = storage

    async def create_continuous_aggregate(self, view_name: str, table_name: str, time_bucket: str, aggregates: List[str]):
        """
        ایجاد `Continuous Aggregate` برای یک جدول داده‌های سری‌زمانی

        Args:
            view_name (str): نام نمای تجمیعی
            table_name (str): نام جدول داده‌های سری‌زمانی
            time_bucket (str): بازه زمانی برای تجمیع (مثلاً '1 hour', '1 day')
            aggregates (List[str]): لیستی از توابع تجمیعی (AVG, SUM, MIN, MAX)
        """
        agg_expressions = ", ".join(aggregates)
        query = f"""
            CREATE MATERIALIZED VIEW {view_name}
            WITH (timescaledb.continuous) AS
            SELECT time_bucket('{time_bucket}', timestamp) AS bucket,
                   {agg_expressions}
            FROM {table_name}
            GROUP BY bucket;
        """

        try:
            logger.info(f"📊 ایجاد Continuous Aggregate `{view_name}` از `{table_name}`...")
            await self.storage.execute_query(query)
            logger.info(f"✅ Continuous Aggregate `{view_name}` ایجاد شد.")
        except Exception as e:
            logger.error(f"❌ خطا در ایجاد Continuous Aggregate: {e}")
            raise

    async def refresh_continuous_aggregate(self, view_name: str, start_time: str, end_time: str):
        """
        بازسازی داده‌های `Continuous Aggregate` در یک بازه مشخص

        Args:
            view_name (str): نام نمای تجمیعی
            start_time (str): زمان شروع بازه (فرمت: 'YYYY-MM-DD HH:MI:SS')
            end_time (str): زمان پایان بازه
        """
        query = f"""
            CALL refresh_continuous_aggregate('{view_name}', '{start_time}', '{end_time}');
        """

        try:
            logger.info(f"♻️ بازسازی داده‌های `{view_name}` برای بازه `{start_time}` - `{end_time}`...")
            await self.storage.execute_query(query)
            logger.info(f"✅ بازسازی `{view_name}` انجام شد.")
        except Exception as e:
            logger.error(f"❌ خطا در بازسازی Continuous Aggregate: {e}")
            raise
