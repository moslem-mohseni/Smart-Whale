import logging
from typing import List, Dict, Optional
from ai.models.language.infrastructure.vector_store.milvus_adapter import MilvusAdapter
from ai.models.language.infrastructure.caching.cache_manager import CacheManager

class VectorSearch:
    """
    این کلاس جستجوی برداری داده‌های پردازشی زبان را مدیریت می‌کند.
    """

    def __init__(self, milvus_adapter: MilvusAdapter, cache_manager: CacheManager):
        self.milvus = milvus_adapter
        self.cache = cache_manager
        logging.info("✅ VectorSearch مقداردهی شد.")

    async def search_vectors(self, collection_name: str, query_vector: List[float], top_k: int = 5) -> Optional[List[Dict]]:
        """
        جستجوی برداری برای یافتن نزدیک‌ترین بردارها به `query_vector`.

        1️⃣ ابتدا بررسی می‌شود که نتیجه‌ی جستجو در `cache_manager` موجود است یا نه.
        2️⃣ در صورت نبود، از Milvus جستجو انجام می‌شود.
        3️⃣ نتیجه در `cache_manager` ذخیره می‌شود تا در درخواست‌های بعدی سریع‌تر بازیابی شود.

        :param collection_name: نام مجموعه‌ی Milvus که داده‌ها در آن ذخیره شده‌اند.
        :param query_vector: بردار مورد جستجو.
        :param top_k: تعداد نتایج برتر که باید بازگردانده شوند.
        :return: لیستی از بردارهای مشابه در صورت یافت شدن.
        """
        cache_key = f"vector_search:{collection_name}:{str(query_vector)}"
        cached_result = await self.cache.get_cached_result(cache_key)
        if cached_result:
            logging.info(f"📥 نتیجه‌ی جستجو از کش دریافت شد: {cache_key}")
            return cached_result

        # اجرای جستجو در Milvus
        search_results = await self.milvus.search_vectors(collection_name, query_vector, top_k)
        if search_results:
            logging.info(f"🔍 جستجوی برداری انجام شد و {len(search_results)} نتیجه یافت شد.")
            await self.cache.cache_result(cache_key, search_results, ttl=300)  # ذخیره در کش به مدت ۵ دقیقه
            return search_results

        logging.warning(f"⚠️ هیچ بردار مشابهی برای جستجوی داده‌شده یافت نشد.")
        return None
