from infrastructure.clickhouse.adapters.clickhouse_adapter import ClickHouseAdapter
from infrastructure.clickhouse.adapters.retry_mechanism import RetryHandler
from typing import List, Dict, Any, Optional

class ClickHouseManager:
    """
    مدیریت ذخیره‌سازی داده‌های پردازشی در ClickHouse.
    """

    def __init__(self):
        self.adapter = ClickHouseAdapter()
        self.retry_handler = RetryHandler()

    async def connect(self) -> None:
        """ اتصال به ClickHouse """
        await self.adapter.connect()

    async def insert(self, table: str, data: Dict[str, Any]) -> bool:
        """
        درج یک ردیف داده در ClickHouse.

        :param table: نام جدول موردنظر
        :param data: داده‌ای که باید درج شود (فرمت دیکشنری)
        :return: True در صورت موفقیت، False در صورت خطا
        """
        columns = ", ".join(data.keys())
        values = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in data.values())
        query = f"INSERT INTO {table} ({columns}) VALUES ({values})"

        return await self.retry_handler.execute_with_retry(self.adapter.execute, query)

    async def batch_insert(self, table: str, data_list: List[Dict[str, Any]]) -> bool:
        """
        درج دسته‌ای داده‌ها در ClickHouse.

        :param table: نام جدول موردنظر
        :param data_list: لیستی از دیکشنری‌های داده‌ای برای درج
        :return: True در صورت موفقیت، False در صورت خطا
        """
        if not data_list:
            return False

        columns = ", ".join(data_list[0].keys())
        values_list = [
            "(" + ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in data.values()) + ")"
            for data in data_list
        ]
        values = ", ".join(values_list)
        query = f"INSERT INTO {table} ({columns}) VALUES {values}"

        return await self.retry_handler.execute_with_retry(self.adapter.execute, query)

    async def query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        اجرای یک کوئری و دریافت نتایج.

        :param query: متن کوئری SQL
        :return: لیستی از دیکشنری‌های داده یا None در صورت خطا
        """
        return await self.retry_handler.execute_with_retry(self.adapter.fetch_all, query)

    async def close(self) -> None:
        """ قطع اتصال از ClickHouse """
        await self.adapter.disconnect()
