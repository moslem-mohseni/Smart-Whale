from infrastructure.redis.service.cache_service import CacheService
from ai.models.language.core.optimizer.quantum_compressor import QuantumCompressor
from ai.models.language.core.processor.feature_extractor import FeatureExtractor
from typing import Dict, Any, List, Optional


class L2Cache:
    """
    این کلاس مسئول مدیریت حافظه‌ی میان‌مدت (`L2 Cache`) برای ذخیره‌ی اطلاعات زمینه‌ای مکالمه است.
    پیام‌ها در `Redis` نگه‌داری شده و برای پردازش‌های میان‌مدت ذخیره می‌شوند.
    """

    def __init__(self, max_size: int = 50, expiration_time: int = 1800):
        """
        مقداردهی اولیه `L2 Cache` با استفاده از `Redis` و بهینه‌سازی‌های لازم.
        :param max_size: حداکثر تعداد پیام‌هایی که در `L2 Cache` نگه داشته می‌شود.
        :param expiration_time: مدت زمان نگهداری پیام‌ها در کش (بر حسب ثانیه، پیش‌فرض ۳۰ دقیقه).
        """
        self.max_size = max_size
        self.expiration_time = expiration_time
        self.cache_service = CacheService()
        self.quantum_compressor = QuantumCompressor()
        self.feature_extractor = FeatureExtractor()

    async def store_message(self, user_id: str, chat_id: str, message: str) -> None:
        """
        ذخیره‌ی پیام جدید در `L2 Cache` در `Redis` با پردازش‌های بهینه‌سازی.
        """
        key = f"l2_cache:{user_id}:{chat_id}"
        messages = await self.cache_service.get(key) or []

        # استخراج ویژگی‌های کلیدی پیام
        important_features = self.feature_extractor.extract_features(message)

        messages.append({"text": message, "features": important_features})

        if len(messages) > self.max_size:
            messages.pop(0)  # حذف قدیمی‌ترین پیام در صورت عبور از حد مجاز

        # فشرده‌سازی داده‌ها قبل از ذخیره‌سازی
        compressed_messages = self.quantum_compressor.compress_messages([msg["text"] for msg in messages])

        await self.cache_service.set(key, compressed_messages, ttl=self.expiration_time)

    async def retrieve_messages(self, user_id: str, chat_id: str) -> List[Dict[str, Any]]:
        """
        بازیابی پیام‌های ذخیره‌شده در `L2 Cache` برای یک کاربر مشخص.
        """
        key = f"l2_cache:{user_id}:{chat_id}"
        compressed_data = await self.cache_service.get(key)

        if compressed_data:
            decompressed_messages = self.quantum_compressor.decompress_messages(compressed_data)
            return [{"text": msg} for msg in decompressed_messages]

        return []

    async def batch_store_messages(self, user_id: str, chat_id: str, messages: List[str]) -> None:
        """
        ذخیره‌ی `Batch` پیام‌ها در `L2 Cache` در صورت بازیابی از `L3 Cache`.
        """
        key = f"l2_cache:{user_id}:{chat_id}"
        compressed_messages = self.quantum_compressor.compress_messages(messages)
        await self.cache_service.set(key, compressed_messages, ttl=self.expiration_time)

    async def get_message_count(self, user_id: str, chat_id: str) -> int:
        """
        دریافت تعداد پیام‌های ذخیره‌شده در `L2 Cache`.
        """
        key = f"l2_cache:{user_id}:{chat_id}"
        compressed_data = await self.cache_service.get(key)

        if compressed_data:
            decompressed_messages = self.quantum_compressor.decompress_messages(compressed_data)
            return len(decompressed_messages)

        return 0

    async def clear_cache(self, user_id: str, chat_id: str) -> None:
        """
        پاک‌سازی `L2 Cache` برای یک چت مشخص‌شده.
        """
        key = f"l2_cache:{user_id}:{chat_id}"
        await self.cache_service.delete(key)

    async def process(self, user_id: str, chat_id: str, message: str) -> Dict[str, Any]:
        """
        ذخیره و بازیابی پیام‌های میان‌مدت کاربر.
        """
        await self.store_message(user_id, chat_id, message)
        messages = await self.retrieve_messages(user_id, chat_id)

        return {
            "user_id": user_id,
            "chat_id": chat_id,
            "context_messages": messages,
        }


# تست اولیه ماژول
if __name__ == "__main__":
    import asyncio


    async def test_l2_cache():
        l2_cache = L2Cache(max_size=10, expiration_time=1200)  # ذخیره‌ی پیام‌ها برای ۲۰ دقیقه

        user_id = "user_456"
        chat_id = "chat_789"
        messages = [
            "سلام، می‌توانی درباره هوش مصنوعی توضیح بدهی؟",
            "چه تفاوتی بین یادگیری عمیق و یادگیری ماشین وجود دارد؟",
            "مدل GPT-4 چگونه کار می‌کند؟",
            "بهترین روش برای پردازش متن چیست؟",
            "چه الگوریتم‌هایی برای پردازش زبان طبیعی وجود دارد؟"
        ]

        for msg in messages:
            result = await l2_cache.process(user_id, chat_id, msg)
            print("\n🔹 Updated L2 Cache:")
            print(result)

        print("\n🔹 Retrieving Context Messages After Expiration Check:")
        retrieved_messages = await l2_cache.retrieve_messages(user_id, chat_id)
        print(retrieved_messages)


    asyncio.run(test_l2_cache())
