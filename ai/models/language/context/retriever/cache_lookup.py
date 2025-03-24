import logging
from typing import Optional, Dict
from ai.models.language.context.memory.l1_cache import L1Cache
from ai.models.language.context.memory.l2_cache import L2Cache
from ai.models.language.context.retriever.vector_search import RetrieverVectorSearch

class CacheLookup:
    """
    این کلاس بررسی می‌کند که آیا داده‌های مکالمه‌ای در کش موجود هستند یا باید از روش‌های بازیابی دیگر استفاده شود.
    """

    def __init__(self):
        self.l1_cache = L1Cache()
        self.l2_cache = L2Cache()
        self.vector_search = RetrieverVectorSearch()
        logging.info("✅ CacheLookup مقداردهی شد.")

    async def retrieve_from_cache(self, user_id: str, chat_id: str, query: str) -> Optional[Dict]:
        """
        تلاش برای یافتن اطلاعات مرتبط از کش (`L1Cache` و `L2Cache`).

        1️⃣ ابتدا `L1Cache` بررسی می‌شود.
        2️⃣ اگر در `L1` نبود، `L2Cache` بررسی می‌شود.
        3️⃣ اگر در `L2` هم نبود، `RetrieverVectorSearch` بررسی می‌شود.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param query: پیام مورد جستجو
        :return: داده‌های مکالمه‌ای ذخیره‌شده (در صورت وجود)
        """

        # بررسی در `L1Cache`
        context = await self.l1_cache.retrieve_messages(user_id, chat_id)
        if context:
            logging.info(f"📥 اطلاعات از `L1Cache` بازیابی شد: {context}")
            return context

        # بررسی در `L2Cache`
        context = await self.l2_cache.retrieve_messages(user_id, chat_id)
        if context:
            logging.info(f"📥 اطلاعات از `L2Cache` بازیابی شد: {context}")
            return context

        # بررسی در `Vector Search`
        related_messages = await self.vector_search.find_related_messages(query, top_n=3)
        if related_messages:
            logging.info(f"🔍 اطلاعات از `Vector Search` بازیابی شد: {related_messages}")
            return {"related_messages": related_messages}

        logging.warning(f"⚠️ اطلاعات برای کاربر {user_id} و چت {chat_id} یافت نشد.")
        return None

    async def store_in_cache(self, user_id: str, chat_id: str, message: Dict):
        """
        ذخیره‌ی داده‌های مکالمه‌ای در کش‌های `L1Cache` و `L2Cache`.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param message: داده‌ی مکالمه‌ای که باید ذخیره شود
        """
        await self.l1_cache.store_message(user_id, chat_id, message)
        await self.l2_cache.store_message(user_id, chat_id, message)

        logging.info(f"✅ اطلاعات در کش ذخیره شد. [User: {user_id} | Chat: {chat_id}]")
