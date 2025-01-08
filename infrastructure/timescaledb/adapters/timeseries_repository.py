# infrastructure/timescaledb/adapters/timeseries_repository.py

from typing import List, TypeVar, Optional, Dict, Any
from datetime import datetime
import logging
from .repository import Repository
from ..domain.models import TimeSeriesData
from ..domain.value_objects import TimeRange
from ...interfaces import StorageInterface, OperationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TimeSeriesRepository(Repository[T]):
    """
    پیاده‌سازی Repository برای داده‌های سری زمانی

    این کلاس عملیات تخصصی مورد نیاز برای کار با داده‌های سری زمانی را
    به قابلیت‌های پایه Repository اضافه می‌کند.
    """

    def __init__(self, storage: StorageInterface, table_name: str):
        """
        راه‌اندازی Repository

        Args:
            storage: رابط ذخیره‌سازی داده
            table_name: نام جدول داده‌های سری زمانی
        """
        self.storage = storage
        self.table_name = table_name

    async def add(self, entity: T) -> T:
        """
        افزودن یک موجودیت جدید

        Args:
            entity: موجودیت مورد نظر برای ذخیره‌سازی

        Returns:
            موجودیت ذخیره شده

        Raises:
            OperationError: در صورت بروز خطا در ذخیره‌سازی
        """
        try:
            query = f"""
                INSERT INTO {self.table_name} 
                (id, timestamp, value, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING *
            """
            params = [
                getattr(entity, 'id'),
                getattr(entity, 'timestamp'),
                getattr(entity, 'value'),
                getattr(entity, 'metadata', {})
            ]
            result = await self.storage.execute(query, params)
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error adding entity: {str(e)}")
            raise OperationError(f"Failed to add entity: {str(e)}")

    async def get(self, id: str) -> Optional[T]:
        """
        بازیابی یک موجودیت با شناسه

        Args:
            id: شناسه موجودیت

        Returns:
            موجودیت یافت شده یا None
        """
        try:
            query = f"SELECT * FROM {self.table_name} WHERE id = $1"
            result = await self.storage.execute(query, [id])
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting entity {id}: {str(e)}")
            raise OperationError(f"Failed to get entity {id}: {str(e)}")

    async def update(self, entity: T) -> Optional[T]:
        """
        به‌روزرسانی یک موجودیت

        Args:
            entity: موجودیت با اطلاعات جدید

        Returns:
            موجودیت به‌روز شده
        """
        try:
            query = f"""
                UPDATE {self.table_name}
                SET timestamp = $2, value = $3, metadata = $4
                WHERE id = $1
                RETURNING *
            """
            params = [
                getattr(entity, 'id'),
                getattr(entity, 'timestamp'),
                getattr(entity, 'value'),
                getattr(entity, 'metadata', {})
            ]
            result = await self.storage.execute(query, params)
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error updating entity: {str(e)}")
            raise OperationError(f"Failed to update entity: {str(e)}")

    async def delete(self, id: str) -> bool:
        """
        حذف یک موجودیت

        Args:
            id: شناسه موجودیت

        Returns:
            True اگر حذف موفق بوده
        """
        try:
            query = f"DELETE FROM {self.table_name} WHERE id = $1"
            result = await self.storage.execute(query, [id])
            return bool(result)
        except Exception as e:
            logger.error(f"Error deleting entity {id}: {str(e)}")
            raise OperationError(f"Failed to delete entity {id}: {str(e)}")

    async def get_range(self, time_range: TimeRange) -> List[T]:
        """
        بازیابی داده‌ها در یک بازه زمانی

        Args:
            time_range: بازه زمانی مورد نظر

        Returns:
            لیست موجودیت‌های یافت شده
        """
        try:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE timestamp BETWEEN $1 AND $2
                ORDER BY timestamp
            """
            params = [time_range.start, time_range.end]
            return await self.storage.execute(query, params)
        except Exception as e:
            logger.error(f"Error getting data range: {str(e)}")
            raise OperationError(f"Failed to get data range: {str(e)}")

    async def get_aggregated(self,
                             time_range: TimeRange,
                             interval: str,
                             aggregations: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        بازیابی داده‌های تجمیع شده

        Args:
            time_range: بازه زمانی
            interval: فاصله زمانی تجمیع (مثلاً '1 hour')
            aggregations: توابع تجمیع مورد نظر (مثلاً {'value': 'avg'})

        Returns:
            لیست نتایج تجمیع شده
        """
        try:
            if not aggregations:
                aggregations = {'value': 'avg'}

            agg_functions = [
                f"{func}({column}) as {column}_{func}"
                for column, func in aggregations.items()
            ]

            query = f"""
                SELECT 
                    time_bucket($1, timestamp) as bucket,
                    {', '.join(agg_functions)}
                FROM {self.table_name}
                WHERE timestamp BETWEEN $2 AND $3
                GROUP BY bucket
                ORDER BY bucket
            """
            params = [interval, time_range.start, time_range.end]
            return await self.storage.execute(query, params)
        except Exception as e:
            logger.error(f"Error getting aggregated data: {str(e)}")
            raise OperationError(f"Failed to get aggregated data: {str(e)}")

    async def get_latest(self, limit: int = 1) -> List[T]:
        """
        بازیابی آخرین داده‌ها

        Args:
            limit: تعداد رکوردهای مورد نیاز

        Returns:
            لیست آخرین داده‌ها
        """
        try:
            query = f"""
                SELECT * FROM {self.table_name}
                ORDER BY timestamp DESC
                LIMIT $1
            """
            return await self.storage.execute(query, [limit])
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            raise OperationError(f"Failed to get latest data: {str(e)}")