from infrastructure.redis.service.cache_service import CacheService
from ai.models.language.core.processor.text_normalizer import TextNormalizer
from ai.models.language.core.processor.feature_extractor import FeatureExtractor
from typing import Dict, Any, List, Optional

class L1Cache:
    """
    این کلاس مسئول مدیریت حافظه‌ی کوتاه‌مدت (`L1 Cache`) برای ذخیره‌ی سریع مکالمات اخیر کاربران است.
    پیام‌ها در `Redis` نگه‌داری شده و هنگام عبور از ۲۰ پیام، پیام‌های قدیمی حذف می‌شوند.
    """

    def __init__(self, max_size: int = 20, expiration_time: int = 300):
        """
        مقداردهی اولیه `L1 Cache` با استفاده از `Redis` و پردازش‌های بهینه.
        :param max_size: حداکثر تعداد پیام‌هایی که در حافظه‌ی `L1 Cache` نگه داشته می‌شود.
        :param expiration_time: مدت زمان نگهداری پیام‌ها در کش (بر حسب ثانیه).
        """
        self.max_size = max_size
        self.expiration_time = expiration_time
        self.cache_service = CacheService()
        self.text_normalizer = TextNormalizer()
        self.feature_extractor = FeatureExtractor()

    async def store_message(self, user_id: str, chat_id: str, message: str) -> None:
        """
        ذخیره‌ی پیام جدید در `L1 Cache` در `Redis` با پیش‌پردازش‌های لازم.
        """
        key = f"l1_cache:{user_id}:{chat_id}"
        messages = await self.cache_service.get(key) or []

        # نرمال‌سازی متن پیام
        normalized_message = self.text_normalizer.normalize(message)

        # استخراج ویژگی‌های کلیدی برای بهینه‌سازی داده‌ها
        important_features = self.feature_extractor.extract_features(normalized_message)

        messages.append({"text": normalized_message, "features": important_features})
        if len(messages) > self.max_size:
            messages.pop(0)  # حذف قدیمی‌ترین پیام در صورت عبور از حد مجاز

        await self.cache_service.set(key, messages, ttl=self.expiration_time)

    async def retrieve_messages(self, user_id: str, chat_id: str) -> List[Dict[str, Any]]:
        """
        بازیابی پیام‌های ذخیره‌شده در `L1 Cache` برای یک کاربر مشخص.
        """
        key = f"l1_cache:{user_id}:{chat_id}"
        return await self.cache_service.get(key) or []

    async def clear_cache(self, user_id: str, chat_id: str) -> None:
        """
        پاک‌سازی `L1 Cache` برای یک چت مشخص‌شده.
        """
        key = f"l1_cache:{user_id}:{chat_id}"
        await self.cache_service.delete(key)

    async def process(self, user_id: str, chat_id: str, message: str) -> Dict[str, Any]:
        """
        ذخیره و بازیابی پیام‌های کوتاه‌مدت کاربر.
        """
        await self.store_message(user_id, chat_id, message)
        messages = await self.retrieve_messages(user_id, chat_id)

        return {
            "user_id": user_id,
            "chat_id": chat_id,
            "recent_messages": messages,
        }


# تست اولیه ماژول
if __name__ == "__main__":
    import asyncio

    async def test_l1_cache():
        l1_cache = L1Cache(max_size=5, expiration_time=180)  # ذخیره‌ی پیام‌ها برای ۳ دقیقه

        user_id = "user_123"
        chat_id = "chat_456"
        messages = [
            "سلام، امروز چه خبر؟",
            "من درباره‌ی یادگیری ماشین کنجکاو هستم.",
            "می‌توانی در مورد شبکه‌های عصبی توضیح دهی؟",
            "تفاوت بین RNN و Transformer چیست؟",
            "چگونه می‌توان از `Fine-Tuning` در یادگیری عمیق استفاده کرد؟"
        ]

        for msg in messages:
            result = await l1_cache.process(user_id, chat_id, msg)
            print("\n🔹 Updated L1 Cache:")
            print(result)

        print("\n🔹 Retrieving Messages After Expiration Check:")
        retrieved_messages = await l1_cache.retrieve_messages(user_id, chat_id)
        print(retrieved_messages)

    asyncio.run(test_l1_cache())
