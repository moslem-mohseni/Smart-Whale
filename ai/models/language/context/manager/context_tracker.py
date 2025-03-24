import logging
from typing import Optional, Dict
from ai.models.language.context.memory.l1_cache import L1Cache
from ai.models.language.context.memory.l2_cache import L2Cache
from ai.models.language.context.memory.l3_cache import L3Cache
from ai.models.language.context.retriever import data_aggregator
from ai.models.language.context.manager.state_manager import StateManager
from ai.models.language.context.manager.session_handler import SessionHandler

class ContextTracker:
    """
    این کلاس وظیفه‌ی رهگیری جریان مکالمه و مدیریت داده‌های زمینه‌ای در هر گفتگو را بر عهده دارد.
    """

    def __init__(self):
        self.l1_cache = L1Cache()
        self.l2_cache = L2Cache()
        self.l3_cache = L3Cache()
        self.data_aggregator = data_aggregator
        self.state_manager = StateManager()
        self.session_handler = SessionHandler()
        logging.info("✅ ContextTracker مقداردهی شد.")

    async def track_conversation(self, user_id: str, chat_id: str, query: str) -> Optional[Dict]:
        """
        رهگیری و بازیابی داده‌های زمینه‌ای برای مکالمه‌ی کاربر.

        1️⃣ ابتدا داده‌های مکالمه‌ای از `L1Cache` و `L2Cache` بررسی می‌شوند.
        2️⃣ اگر داده‌ای یافت نشد، از `L3Cache` بازیابی می‌شود.
        3️⃣ در صورت عدم وجود داده، از `data_aggregator` درخواست داده می‌شود.
        4️⃣ وضعیت مکالمه با `state_manager` تعیین می‌شود.
        5️⃣ اطلاعات نشست با `session_handler` بررسی و مدیریت می‌شود.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param query: پیام مورد جستجو
        :return: داده‌های زمینه‌ای معتبر (در صورت وجود)
        """

        # بررسی `L1Cache`
        context = await self.l1_cache.retrieve_messages(user_id, chat_id)
        if context:
            logging.info(f"📥 داده‌های مکالمه از `L1Cache` دریافت شد: {context}")
            return context

        # بررسی `L2Cache`
        context = await self.l2_cache.retrieve_messages(user_id, chat_id)
        if context:
            logging.info(f"📥 داده‌های مکالمه از `L2Cache` دریافت شد: {context}")
            return context

        # بررسی `L3Cache`
        context = await self.l3_cache.retrieve_messages(user_id, chat_id)
        if context:
            logging.info(f"📥 داده‌های مکالمه از `L3Cache` دریافت شد: {context}")
            return context

        # دریافت داده از `retriever/`
        aggregated_data = await self.data_aggregator.aggregate_context(user_id, chat_id, query)
        if aggregated_data:
            logging.info(f"🔍 داده‌های پردازش‌شده از `retriever/` دریافت شد: {aggregated_data}")

            # بررسی وضعیت مکالمه
            state = await self.state_manager.determine_state(user_id, chat_id, query)
            logging.info(f"📊 وضعیت مکالمه: {state}")

            # بررسی نشست کاربر
            session_data = await self.session_handler.get_session_data(user_id, chat_id)
            logging.info(f"🔄 اطلاعات نشست: {session_data}")

            return {"context": aggregated_data, "state": state, "session": session_data}

        logging.warning(f"⚠️ هیچ داده‌ی زمینه‌ای برای کاربر {user_id} و چت {chat_id} یافت نشد.")
        return None

    async def update_conversation(self, user_id: str, chat_id: str, new_message: Dict):
        """
        به‌روزرسانی جریان مکالمه و ذخیره در حافظه‌های `L1Cache`, `L2Cache`, `L3Cache`.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param new_message: پیام جدیدی که باید در حافظه ذخیره شود
        """
        await self.l1_cache.store_message(user_id, chat_id, new_message)
        await self.l2_cache.store_message(user_id, chat_id, new_message)
        await self.l3_cache.store_messages(user_id, chat_id, [new_message])

        logging.info(f"✅ مکالمه به‌روز شد و داده در حافظه ذخیره شد. [User: {user_id} | Chat: {chat_id}]")

    async def clear_conversation(self, user_id: str, chat_id: str):
        """
        حذف تمامی داده‌های مربوط به یک مکالمه از حافظه‌های مختلف.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        """
        await self.l1_cache.clear_cache(user_id, chat_id)
        await self.l2_cache.clear_cache(user_id, chat_id)
        await self.l3_cache.clear_cache(user_id, chat_id)

        logging.info(f"🗑️ داده‌های زمینه‌ای مکالمه برای کاربر {user_id} و چت {chat_id} حذف شد.")
