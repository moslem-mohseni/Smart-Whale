from infrastructure.elasticsearch.adapters.elasticsearch_adapter import ElasticsearchAdapter
from infrastructure.elasticsearch.optimization.query_optimizer import QueryOptimizer
from typing import Optional, Dict, Any, List

class ElasticManager:
    """
    مدیریت ذخیره‌سازی و جستجو در Elasticsearch.
    """

    def __init__(self):
        self.adapter = ElasticsearchAdapter()
        self.optimizer = QueryOptimizer()

    async def connect(self) -> None:
        """ اتصال به Elasticsearch """
        await self.adapter.connect()

    async def insert_document(self, index: str, doc_id: str, data: Dict[str, Any]) -> bool:
        """
        درج سند جدید در Elasticsearch.

        :param index: نام ایندکس
        :param doc_id: شناسه سند
        :param data: محتوای سند
        :return: True در صورت موفقیت، False در صورت خطا
        """
        return await self.adapter.insert(index, doc_id, data)

    async def search(self, index: str, query: Dict[str, Any], size: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        جستجوی داده در Elasticsearch.

        :param index: نام ایندکس
        :param query: کوئری جستجو
        :param size: تعداد نتایج موردنظر
        :return: لیستی از اسناد یافت‌شده یا None در صورت خطا
        """
        optimized_query = self.optimizer.optimize_query(query)
        return await self.adapter.search(index, optimized_query, size)

    async def update_document(self, index: str, doc_id: str, update_fields: Dict[str, Any]) -> bool:
        """
        به‌روزرسانی یک سند در Elasticsearch.

        :param index: نام ایندکس
        :param doc_id: شناسه سند
        :param update_fields: فیلدهایی که باید به‌روزرسانی شوند
        :return: True در صورت موفقیت، False در صورت خطا
        """
        return await self.adapter.update(index, doc_id, update_fields)

    async def delete_document(self, index: str, doc_id: str) -> bool:
        """
        حذف یک سند از Elasticsearch.

        :param index: نام ایندکس
        :param doc_id: شناسه سند
        :return: True در صورت حذف موفق، False در غیر این صورت
        """
        return await self.adapter.delete(index, doc_id)

    async def close(self) -> None:
        """ قطع اتصال از Elasticsearch """
        await self.adapter.disconnect()
