import logging
from typing import Dict, List, Optional
from ai.models.language.context.retriever.cache_lookup import CacheLookup
from ai.models.language.context.retriever.vector_search import RetrieverVectorSearch


class DataAggregator:
    """
    این کلاس داده‌های مرتبط با مکالمات را از منابع مختلف (کش و جستجوی برداری) جمع‌آوری کرده و فیلتر می‌کند.
    """

    def __init__(self):
        self.cache_lookup = CacheLookup()
        self.vector_search = RetrieverVectorSearch()
        logging.info("✅ DataAggregator مقداردهی شد.")

    async def aggregate_context(self, user_id: str, chat_id: str, query: str) -> Optional[Dict]:
        """
        جمع‌آوری داده‌های مرتبط با مکالمه از کش و جستجوی برداری.

        1️⃣ ابتدا `cache_lookup` بررسی می‌شود.
        2️⃣ اگر داده‌ای یافت نشد، از `vector_search` استفاده می‌شود.
        3️⃣ سپس داده‌ها ترکیب و فیلتر می‌شوند.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param query: پیام مورد جستجو
        :return: داده‌های ترکیبی مکالمه‌ای (در صورت وجود)
        """

        # بررسی در `CacheLookup`
        cache_data = await self.cache_lookup.retrieve_from_cache(user_id, chat_id, query)
        if cache_data:
            logging.info(f"📥 داده‌های بازیابی‌شده از کش: {cache_data}")
            return {"source": "cache", "data": cache_data}

        # اگر داده‌ای در کش نبود، جستجوی برداری انجام شود
        search_results = await self.vector_search.find_related_messages(query, top_n=5)
        if search_results:
            logging.info(f"🔍 داده‌های بازیابی‌شده از جستجوی برداری: {search_results}")
            return {"source": "vector_search", "data": search_results}

        logging.warning(f"⚠️ هیچ داده‌ی مرتبطی برای کاربر {user_id} و چت {chat_id} یافت نشد.")
        return None
