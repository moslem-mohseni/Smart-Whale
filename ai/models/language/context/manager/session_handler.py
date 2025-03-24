import logging
from typing import Optional, Dict
from ai.models.language.context.memory.l2_cache import L2Cache
from ai.models.language.context.memory.l3_cache import L3Cache
from ai.models.language.context.manager.state_manager import StateManager

class SessionHandler:
    """
    این کلاس مدیریت نشست‌های مکالمه‌ای کاربران را بر عهده دارد و وضعیت تعاملات را ذخیره و بازیابی می‌کند.
    """

    def __init__(self):
        self.l2_cache = L2Cache()
        self.l3_cache = L3Cache()
        self.state_manager = StateManager()
        logging.info("✅ SessionHandler مقداردهی شد.")

    async def start_session(self, user_id: str, chat_id: str):
        """
        ایجاد یک نشست جدید برای کاربر.

        1️⃣ بررسی می‌شود که آیا نشست فعال برای این کاربر وجود دارد یا نه.
        2️⃣ در صورت عدم وجود، نشست جدید ایجاد می‌شود و در `L2Cache` ذخیره می‌شود.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        """
        existing_session = await self.get_session_data(user_id, chat_id)
        if existing_session:
            logging.info(f"✅ نشست قبلی برای کاربر {user_id} در چت {chat_id} یافت شد.")
            return existing_session

        session_data = {
            "user_id": user_id,
            "chat_id": chat_id,
            "messages": [],
            "state": "new",
        }

        await self.l2_cache.store_message(user_id, chat_id, session_data)
        logging.info(f"🆕 نشست جدید برای کاربر {user_id} در چت {chat_id} ایجاد شد.")

    async def get_session_data(self, user_id: str, chat_id: str) -> Optional[Dict]:
        """
        دریافت اطلاعات نشست فعال برای یک کاربر.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :return: داده‌های نشست کاربر (در صورت وجود)
        """
        session_data = await self.l2_cache.retrieve_messages(user_id, chat_id)
        if session_data:
            logging.info(f"📥 داده‌های نشست از `L2Cache` دریافت شد.")
            return session_data

        session_data = await self.l3_cache.retrieve_messages(user_id, chat_id)
        if session_data:
            logging.info(f"📥 داده‌های نشست از `L3Cache` دریافت شد.")
            return session_data

        logging.warning(f"⚠️ هیچ نشست فعالی برای کاربر {user_id} و چت {chat_id} یافت نشد.")
        return None

    async def update_session(self, user_id: str, chat_id: str, new_message: str):
        """
        به‌روزرسانی داده‌های نشست با پیام جدید و مدیریت وضعیت مکالمه.

        1️⃣ پیام جدید در نشست ذخیره می‌شود.
        2️⃣ وضعیت مکالمه در `state_manager` بررسی و به‌روزرسانی می‌شود.
        3️⃣ در صورت نیاز، نشست در `L3Cache` ذخیره می‌شود.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param new_message: پیام جدیدی که باید در نشست ذخیره شود
        """
        session_data = await self.get_session_data(user_id, chat_id)
        if not session_data:
            logging.info(f"🆕 نشست جدید برای کاربر {user_id} ایجاد شد.")
            await self.start_session(user_id, chat_id)
            session_data = await self.get_session_data(user_id, chat_id)

        session_data["messages"].append(new_message)

        # بررسی وضعیت مکالمه و به‌روزرسانی آن
        session_data["state"] = await self.state_manager.determine_state(user_id, chat_id, new_message)

        # ذخیره داده‌های نشست در `L2Cache`
        await self.l2_cache.store_message(user_id, chat_id, session_data)

        # در صورت رسیدن به تعداد مشخصی از پیام‌ها، نشست در `L3Cache` ذخیره شود
        if len(session_data["messages"]) > 10:
            await self.l3_cache.store_messages(user_id, chat_id, [session_data])
            logging.info(f"📦 نشست کاربر {user_id} در `L3Cache` ذخیره شد.")

        logging.info(f"🔄 نشست کاربر {user_id} به‌روزرسانی شد.")

    async def end_session(self, user_id: str, chat_id: str):
        """
        پایان دادن به یک نشست و حذف داده‌های مکالمه.

        1️⃣ داده‌های نشست از `L2Cache` و `L3Cache` حذف می‌شود.
        2️⃣ وضعیت مکالمه به پایان‌یافته تغییر می‌کند.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        """
        await self.l2_cache.clear_cache(user_id, chat_id)
        await self.l3_cache.clear_cache(user_id, chat_id)

        logging.info(f"🗑️ نشست کاربر {user_id} در چت {chat_id} پایان یافت و داده‌ها حذف شدند.")
