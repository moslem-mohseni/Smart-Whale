import logging
from typing import List, Dict, Optional
from ai.models.language.infrastructure.vector_store.vector_search import VectorSearch
from infrastructure.vector_store.service.vector_service import VectorService


class MilvusAdapter:
    """
    این کلاس مدیریت ارتباط با Milvus برای ذخیره و جستجوی بردارهای داده‌های زبانی را بر عهده دارد.
    این ماژول به‌صورت مستقیم از `VectorService` که در `infrastructure/vector_store/` پیاده‌سازی شده، استفاده می‌کند.
    """

    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        logging.info("✅ MilvusAdapter مقداردهی شد و ارتباط با VectorService برقرار شد.")

    async def insert_vectors(self, collection_name: str, vectors: List[List[float]]):
        """
        درج بردارهای جدید در Milvus.

        :param collection_name: نام مجموعه‌ای که بردارها در آن ذخیره می‌شوند.
        :param vectors: لیستی از بردارها که باید ذخیره شوند.
        """
        try:
            await self.vector_service.insert_vectors(collection_name, vectors)
            logging.info(f"✅ {len(vectors)} بردار در مجموعه {collection_name} ذخیره شد.")
        except Exception as e:
            logging.error(f"❌ خطا در ذخیره‌سازی بردارها در Milvus: {e}")

    async def search_vectors(self, collection_name: str, query_vector: List[float], top_k: int = 5) -> Optional[List[Dict]]:
        """
        جستجوی برداری برای یافتن نزدیک‌ترین بردارها به `query_vector`.

        :param collection_name: نام مجموعه‌ای که جستجو در آن انجام می‌شود.
        :param query_vector: بردار مورد جستجو.
        :param top_k: تعداد نتایج برتر که باید بازگردانده شوند.
        :return: لیستی از بردارهای مشابه در صورت یافت شدن.
        """
        try:
            results = await self.vector_service.search_vectors(collection_name, query_vector, top_k)
            logging.info(f"🔍 {len(results)} نتیجه از جستجوی برداری در {collection_name} یافت شد.")
            return results
        except Exception as e:
            logging.error(f"❌ خطا در جستجوی برداری در Milvus: {e}")
            return None

    async def delete_vectors(self, collection_name: str, ids: List[str]):
        """
        حذف بردارهای مشخص‌شده از Milvus.

        :param collection_name: نام مجموعه‌ای که بردارها در آن ذخیره شده‌اند.
        :param ids: لیستی از شناسه‌های بردارهایی که باید حذف شوند.
        """
        try:
            await self.vector_service.delete_vectors(collection_name, ids)
            logging.info(f"🗑 {len(ids)} بردار از مجموعه {collection_name} حذف شد.")
        except Exception as e:
            logging.error(f"❌ خطا در حذف بردارها از Milvus: {e}")
