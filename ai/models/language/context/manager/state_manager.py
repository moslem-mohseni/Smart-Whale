import logging
from typing import Optional, Dict
from ai.models.language.context.memory.l2_cache import L2Cache
from ai.models.language.context.memory.l3_cache import L3Cache

class StateManager:
    """
    این کلاس مسئول مدیریت وضعیت مکالمه، تشخیص مرحله‌ی تعامل و نگهداری داده‌های وضعیت است.
    """

    def __init__(self):
        self.l2_cache = L2Cache()
        self.l3_cache = L3Cache()
        logging.info("✅ StateManager مقداردهی شد.")

    async def determine_state(self, user_id: str, chat_id: str, query: str) -> str:
        """
        تعیین وضعیت مکالمه‌ی کاربر بر اساس تاریخچه‌ی مکالمه.

        1️⃣ بررسی داده‌های مکالمه در `L2Cache` و `L3Cache`.
        2️⃣ تحلیل الگوی مکالمه و شناسایی مرحله‌ی تعامل.
        3️⃣ ذخیره‌ی وضعیت مکالمه برای پردازش‌های بعدی.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param query: پیام ورودی جدید
        :return: وضعیت مکالمه (مانند `greeting`, `inquiry`, `action`, `confirmation`, `closing`)
        """

        # دریافت وضعیت فعلی مکالمه
        current_state = await self.get_state(user_id, chat_id)

        # تحلیل پیام جدید و تعیین مرحله‌ی مکالمه
        new_state = self.analyze_message(query, current_state)

        # ذخیره‌ی وضعیت جدید در `L2Cache`
        await self.l2_cache.store_message(user_id, chat_id, {"state": new_state})

        logging.info(f"📊 وضعیت مکالمه به‌روزرسانی شد: {new_state} [User: {user_id} | Chat: {chat_id}]")
        return new_state

    async def get_state(self, user_id: str, chat_id: str) -> Optional[str]:
        """
        دریافت وضعیت فعلی مکالمه از `L2Cache` یا `L3Cache`.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :return: وضعیت مکالمه در صورت وجود
        """
        state_data = await self.l2_cache.retrieve_messages(user_id, chat_id)
        if state_data and "state" in state_data:
            logging.info(f"📥 وضعیت مکالمه از `L2Cache` دریافت شد: {state_data['state']}")
            return state_data["state"]

        state_data = await self.l3_cache.retrieve_messages(user_id, chat_id)
        if state_data and "state" in state_data:
            logging.info(f"📥 وضعیت مکالمه از `L3Cache` دریافت شد: {state_data['state']}")
            return state_data["state"]

        logging.warning(f"⚠️ هیچ وضعیت مکالمه‌ای برای کاربر {user_id} در چت {chat_id} یافت نشد.")
        return None

    def analyze_message(self, message: str, current_state: Optional[str]) -> str:
        """
        تحلیل پیام جدید و تعیین مرحله‌ی مکالمه.

        :param message: متن پیام ورودی
        :param current_state: وضعیت فعلی مکالمه
        :return: وضعیت جدید مکالمه
        """

        # بررسی کلمات کلیدی و تعیین وضعیت مکالمه
        greeting_keywords = ["سلام", "درود", "صبح بخیر", "سلام علیکم"]
        inquiry_keywords = ["چطور", "چیست", "کجا", "چگونه", "چرا"]
        action_keywords = ["بفرست", "انجام بده", "ثبت کن", "بگیر"]
        confirmation_keywords = ["درسته", "تایید", "اوکی", "قبول دارم"]
        closing_keywords = ["خداحافظ", "ممنون", "بعدا صحبت می‌کنیم"]

        message_lower = message.lower()

        if any(word in message_lower for word in greeting_keywords):
            return "greeting"
        elif any(word in message_lower for word in inquiry_keywords):
            return "inquiry"
        elif any(word in message_lower for word in action_keywords):
            return "action"
        elif any(word in message_lower for word in confirmation_keywords):
            return "confirmation"
        elif any(word in message_lower for word in closing_keywords):
            return "closing"

        # در صورتی که نتوانستیم وضعیت مشخصی تعیین کنیم، از وضعیت فعلی استفاده می‌شود
        return current_state if current_state else "unknown"

    async def update_state(self, user_id: str, chat_id: str, new_state: str):
        """
        به‌روزرسانی وضعیت مکالمه و ذخیره در `L2Cache` و `L3Cache`.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param new_state: وضعیت جدید مکالمه
        """
        await self.l2_cache.store_message(user_id, chat_id, {"state": new_state})
        await self.l3_cache.store_messages(user_id, chat_id, [{"state": new_state}])

        logging.info(f"🔄 وضعیت مکالمه به `{new_state}` تغییر یافت. [User: {user_id} | Chat: {chat_id}]")

    async def clear_state(self, user_id: str, chat_id: str):
        """
        حذف وضعیت مکالمه از حافظه‌های `L2Cache` و `L3Cache`.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        """
        await self.l2_cache.clear_cache(user_id, chat_id)
        await self.l3_cache.clear_cache(user_id, chat_id)

        logging.info(f"🗑️ وضعیت مکالمه برای کاربر {user_id} و چت {chat_id} حذف شد.")
