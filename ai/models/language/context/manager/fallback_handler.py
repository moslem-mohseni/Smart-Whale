import logging
from typing import Optional, Dict
from ai.models.language.context.manager.state_manager import StateManager
from ai.models.language.context.manager.session_handler import SessionHandler
from ai.models.language.context.retriever.vector_search import RetrieverVectorSearch

class FallbackHandler:
    """
    این کلاس وظیفه‌ی مدیریت پاسخ‌های جایگزین را بر عهده دارد.
    در صورت عدم وجود داده‌ی کافی، از روش‌های جایگزین مانند جستجوی برداری استفاده می‌شود.
    """

    def __init__(self):
        self.state_manager = StateManager()
        self.session_handler = SessionHandler()
        self.vector_search = RetrieverVectorSearch()
        logging.info("✅ FallbackHandler مقداردهی شد.")

    async def handle_fallback(self, user_id: str, chat_id: str, query: str) -> Optional[Dict]:
        """
        مدیریت پاسخ‌های جایگزین در صورتی که پاسخ مستقیمی برای پیام ورودی وجود نداشته باشد.

        1️⃣ بررسی وضعیت مکالمه و تعاملات اخیر کاربر.
        2️⃣ تلاش برای یافتن پاسخ‌های مشابه از `vector_search`.
        3️⃣ اگر پاسخی یافت نشد، ارسال پیام عمومی مانند "متوجه نشدم، لطفاً واضح‌تر بپرسید".

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param query: پیام ورودی که نیاز به پاسخ جایگزین دارد
        :return: پاسخ جایگزین در صورت یافتن
        """

        # دریافت وضعیت مکالمه
        current_state = await self.state_manager.get_state(user_id, chat_id)
        logging.info(f"📊 وضعیت مکالمه در `FallbackHandler`: {current_state}")

        # جستجوی برداری برای یافتن پاسخ‌های مشابه
        similar_responses = await self.vector_search.find_related_messages(query, top_n=3)
        if similar_responses:
            logging.info(f"🔍 پاسخ‌های جایگزین از `vector_search` یافت شد: {similar_responses}")
            return {"fallback_response": similar_responses}

        # اگر هیچ پاسخی پیدا نشد، پیام عمومی ارسال شود
        logging.warning(f"⚠️ هیچ پاسخ جایگزینی برای پیام `{query}` یافت نشد. ارسال پاسخ عمومی.")
        return {"fallback_response": ["متوجه نشدم، لطفاً واضح‌تر بپرسید!"]}

    async def log_fallback_case(self, user_id: str, chat_id: str, query: str):
        """
        ثبت مواردی که نیاز به پاسخ جایگزین داشته‌اند، برای بهبود مدل‌های آینده.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param query: پیام ورودی که پاسخی برای آن یافت نشد
        """
        logging.info(f"📝 مورد `fallback` ثبت شد: [User: {user_id} | Chat: {chat_id} | Query: {query}]")
