import logging
from ai.models.language.context.memory.l1_cache import L1Cache
from ai.models.language.context.memory.l2_cache import L2Cache
from ai.models.language.context.memory.l3_cache import L3Cache
from ai.models.language.context.manager.session_handler import SessionHandler

class UpdatePolicy:
    """
    این کلاس وظیفه‌ی مدیریت سیاست‌های به‌روزرسانی داده‌های مکالمه‌ای در کش و حافظه را بر عهده دارد.
    """

    def __init__(self):
        self.l1_cache = L1Cache()
        self.l2_cache = L2Cache()
        self.l3_cache = L3Cache()
        self.session_handler = SessionHandler()
        logging.info("✅ UpdatePolicy مقداردهی شد.")

    async def enforce_policies(self, user_id: str, chat_id: str):
        """
        اجرای سیاست‌های حذف و بروزرسانی داده‌ها در سطوح مختلف کش و حافظه.

        1️⃣ بررسی تعداد پیام‌های ذخیره‌شده در `L1Cache` و حذف پیام‌های قدیمی در صورت نیاز.
        2️⃣ در صورت رسیدن تعداد پیام‌ها به حد مشخص، انتقال به `L3Cache`.
        3️⃣ بررسی وضعیت نشست و در صورت غیرفعال بودن، حذف نشست از `L2Cache`.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        """
        # بررسی تعداد پیام‌های ذخیره‌شده در `L1Cache`
        messages = await self.l1_cache.retrieve_messages(user_id, chat_id)
        if messages and len(messages) > 10:
            await self.l1_cache.clear_cache(user_id, chat_id)
            logging.info(f"🗑️ پیام‌های قدیمی از `L1Cache` حذف شدند.")

        # انتقال داده‌های مکالمه‌ای به `L3Cache` در صورت رسیدن به حد مشخص
        if messages and len(messages) > 20:
            await self.l3_cache.store_messages(user_id, chat_id, messages)
            await self.l2_cache.clear_cache(user_id, chat_id)
            logging.info(f"📦 داده‌های مکالمه‌ای به `L3Cache` منتقل شدند.")

        # بررسی وضعیت نشست کاربر و حذف نشست‌های غیرفعال
        session_data = await self.session_handler.get_session_data(user_id, chat_id)
        if not session_data:
            await self.l2_cache.clear_cache(user_id, chat_id)
            logging.info(f"❌ نشست غیرفعال کاربر {user_id} حذف شد.")

    async def update_cache(self, user_id: str, chat_id: str, new_message: str):
        """
        بروزرسانی کش مکالمه و انتقال داده‌های قدیمی در صورت نیاز.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param new_message: پیام جدیدی که باید ذخیره شود
        """
        await self.l1_cache.store_message(user_id, chat_id, new_message)
        await self.l2_cache.store_message(user_id, chat_id, new_message)

        # بررسی حجم داده‌های ذخیره‌شده و انتقال به `L3Cache`
        messages = await self.l2_cache.retrieve_messages(user_id, chat_id)
        if messages and len(messages) > 15:
            await self.l3_cache.store_messages(user_id, chat_id, messages)
            await self.l2_cache.clear_cache(user_id, chat_id)
            logging.info(f"📦 داده‌های مکالمه‌ای به `L3Cache` منتقل شدند.")

    async def clear_old_data(self, user_id: str, chat_id: str):
        """
        حذف داده‌های قدیمی از تمامی حافظه‌ها در صورت عدم فعالیت طولانی‌مدت.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        """
        await self.l1_cache.clear_cache(user_id, chat_id)
        await self.l2_cache.clear_cache(user_id, chat_id)
        await self.l3_cache.clear_cache(user_id, chat_id)

        logging.info(f"🗑️ تمامی داده‌های قدیمی برای کاربر {user_id} و چت {chat_id} حذف شدند.")
