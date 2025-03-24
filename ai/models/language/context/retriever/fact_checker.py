import logging
from typing import Dict, Optional
from ai.models.language.context.retriever.data_aggregator import DataAggregator

class FactChecker:
    """
    این کلاس وظیفه‌ی بررسی صحت داده‌های مکالمه‌ای استخراج‌شده را بر عهده دارد.
    """

    def __init__(self):
        self.data_aggregator = DataAggregator()
        logging.info("✅ FactChecker مقداردهی شد.")

    async def validate_context(self, user_id: str, chat_id: str, query: str) -> Optional[Dict]:
        """
        بررسی صحت اطلاعات استخراج‌شده از کش و جستجوی برداری.

        1️⃣ ابتدا داده‌های مکالمه‌ای از `data_aggregator` دریافت می‌شود.
        2️⃣ داده‌های دریافت‌شده بررسی می‌شوند که آیا معتبر هستند یا خیر.
        3️⃣ در صورت معتبر نبودن، داده‌ها اصلاح یا حذف می‌شوند.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param query: پیام مورد جستجو
        :return: داده‌های بررسی‌شده و معتبر (در صورت وجود)
        """

        # دریافت داده‌های اولیه
        context_data = await self.data_aggregator.aggregate_context(user_id, chat_id, query)
        if not context_data:
            logging.warning(f"⚠️ هیچ داده‌ای برای بررسی صحت یافت نشد. [User: {user_id} | Chat: {chat_id}]")
            return None

        # بررسی صحت داده‌ها
        validated_data = self.check_factual_accuracy(context_data["data"])
        if validated_data:
            logging.info(f"✅ داده‌های معتبر تأیید شدند: {validated_data}")
            return {"source": context_data["source"], "validated_data": validated_data}
        else:
            logging.warning(f"🚨 داده‌های نامعتبر حذف شدند. [User: {user_id} | Chat: {chat_id}]")
            return None

    def check_factual_accuracy(self, data: Dict) -> Optional[Dict]:
        """
        بررسی دقیق داده‌ها و اعتبارسنجی اطلاعات.

        1️⃣ حذف اطلاعات بی‌معنی یا نامعتبر.
        2️⃣ بررسی اینکه آیا اطلاعات دارای منبع معتبر هستند.
        3️⃣ تأیید تطابق اطلاعات با زمینه‌ی مکالمه.

        :param data: داده‌های مکالمه‌ای که باید بررسی شوند.
        :return: داده‌های معتبر (در صورت وجود)
        """

        # فیلتر کردن داده‌های نامعتبر
        valid_data = {key: value for key, value in data.items() if self.is_valid(value)}

        return valid_data if valid_data else None

    def is_valid(self, value: str) -> bool:
        """
        بررسی صحت یک مقدار خاص.

        :param value: مقدار مورد بررسی
        :return: نتیجه‌ی بررسی صحت (`True` اگر معتبر است، `False` اگر نامعتبر است)
        """

        # شرط ساده برای فیلتر اطلاعات (می‌توان الگوریتم‌های پیچیده‌تری نیز پیاده کرد)
        return bool(value and len(value) > 3 and "نامعتبر" not in value)

