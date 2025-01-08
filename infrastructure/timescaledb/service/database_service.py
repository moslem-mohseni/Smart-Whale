# infrastructure/timescaledb/service/database_service.py

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from ..config.settings import TimescaleDBConfig
from ..domain.models import TimeSeriesData, TableSchema
from ...interfaces import StorageInterface, ConnectionError, OperationError

logger = logging.getLogger(__name__)


class TimescaleDBService:
    """سرویس مدیریت ارتباط با TimescaleDB"""

    def __init__(self, config: TimescaleDBConfig):
        self.config = config
        self._storage: Optional[StorageInterface] = None

    async def initialize(self) -> None:
        """راه‌اندازی اولیه سرویس"""
        storage = await self.get_storage()
        await storage.connect()

    async def shutdown(self) -> None:
        """خاتمه سرویس"""
        if self._storage:
            await self._storage.disconnect()

    async def get_storage(self) -> StorageInterface:
        """دریافت اتصال به دیتابیس"""
        if not self._storage or not await self._storage.is_connected():
            self._storage = self._create_storage()
            await self._storage.connect()
        return self._storage

    def _create_storage(self) -> StorageInterface:
        """ایجاد نمونه جدید از کلاس Storage"""
        from ..storage.timescaledb_storage import TimescaleDBStorage
        return TimescaleDBStorage(self.config)

    async def store_time_series_data(self, data: TimeSeriesData) -> None:
        """ذخیره‌سازی داده‌های سری زمانی"""
        storage = await self.get_storage()
        query = """
            INSERT INTO time_series_data (id, timestamp, value, metadata)
            VALUES ($1, $2, $3, $4)
        """
        await storage.execute(query, [data.id, data.timestamp, data.value, data.metadata])

    async def get_time_series_data(self, id: str) -> Optional[Dict]:
        """بازیابی داده‌های سری زمانی با شناسه"""
        storage = await self.get_storage()
        query = "SELECT * FROM time_series_data WHERE id = $1"
        result = await storage.execute(query, [id])
        return result[0] if result else None

    async def aggregate_time_series(self, metric: str, interval: str,
                                    start_time: datetime, end_time: datetime) -> List[Dict]:
        """تجمیع داده‌های سری زمانی"""
        storage = await self.get_storage()
        query = f"""
            SELECT time_bucket($1, timestamp) as bucket,
                   avg({metric}) as avg_value
            FROM time_series_data
            WHERE timestamp BETWEEN $2 AND $3
            GROUP BY bucket
            ORDER BY bucket
        """
        return await storage.execute(query, [interval, start_time, end_time])

    async def create_continuous_aggregate(self, view_name: str, table_name: str,
                                          interval: str, aggregates: List[str]) -> None:
        """ایجاد تجمیع مستمر"""
        storage = await self.get_storage()
        agg_expressions = ", ".join(aggregates)
        query = f"""
            CREATE MATERIALIZED VIEW {view_name}
            WITH (timescaledb.continuous) AS
            SELECT time_bucket('{interval}', timestamp) AS bucket,
                   {agg_expressions}
            FROM {table_name}
            GROUP BY bucket
        """
        await storage.execute(query)

    async def set_retention_policy(self, table_name: str, interval: str) -> None:
        """تنظیم سیاست نگهداری داده"""
        storage = await self.get_storage()
        query = f"""
            ALTER TABLE {table_name} SET (
                timescaledb.drop_after = '{interval}'::interval
            )
        """
        await storage.execute(query)

    async def set_compression_policy(self, table_name: str, segment_by: str,
                                     order_by: str) -> None:
        """تنظیم سیاست فشرده‌سازی"""
        storage = await self.get_storage()
        query = f"""
            ALTER TABLE {table_name} SET (
                timescaledb.compress = true,
                timescaledb.compress_segmentby = '{segment_by}',
                timescaledb.compress_orderby = '{order_by}'
            )
        """
        await storage.execute(query)

    async def create_hypertable(self, schema: TableSchema,
                                partition_interval: str) -> None:
        """ایجاد جدول و تبدیل به hypertable"""
        storage = await self.get_storage()

        # ایجاد جدول
        await storage.create_table(schema.name, schema.columns)

        # تبدیل به hypertable
        if schema.time_column:
            await storage.create_hypertable(
                schema.name,
                schema.time_column,
                interval=partition_interval
            )

        # ایجاد ایندکس‌ها
        if schema.indexes:
            for name, definition in schema.indexes.items():
                query = f"CREATE INDEX IF NOT EXISTS {name} ON {schema.name} {definition}"
                await storage.execute(query)