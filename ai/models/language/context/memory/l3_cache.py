from infrastructure.timescaledb.service.database_service import DatabaseService
from ai.models.language.core.optimizer.retrieval_optimizer import RetrievalOptimizer
from ai.models.language.core.optimizer.quantum_compressor import QuantumCompressor
from typing import Dict, Any, List, Optional
from datetime import datetime


class L3Cache:
    """
    این کلاس مسئول مدیریت حافظه‌ی بلندمدت (`L3 Cache`) برای ذخیره‌ی اطلاعات مکالمه در `TimescaleDB` است.
    مکالمات بعد از `۱۰ پیام` یا `۳۰ ثانیه` از `L2 Cache` به `L3 Cache` منتقل می‌شوند.
    """

    def __init__(self):
        """
        مقداردهی اولیه `L3 Cache` با استفاده از `TimescaleDB` و ابزارهای بهینه‌سازی.
        """
        self.db_service = DatabaseService()
        self.retrieval_optimizer = RetrievalOptimizer()
        self.quantum_compressor = QuantumCompressor()

    async def store_messages(self, user_id: str, chat_id: str, messages: List[str]) -> None:
        """
        ذخیره‌ی `Batch` داده‌های مکالمه‌ای در `L3 Cache` (`TimescaleDB`).
        """
        timestamp = datetime.utcnow()

        # فشرده‌سازی پیام‌ها برای بهینه‌سازی مصرف دیتابیس
        compressed_messages = self.quantum_compressor.compress_messages(messages)

        formatted_messages = [{
            "user_id": user_id,
            "chat_id": chat_id,
            "timestamp": timestamp,
            "compressed_message": compressed_messages
        }]

        await self.db_service.batch_store_time_series_data("conversation_history", formatted_messages)

    async def retrieve_messages(self, user_id: str, chat_id: str, time_range: Optional[Dict[str, datetime]] = None) -> \
    List[str]:
        """
        بازیابی مکالمات ذخیره‌شده در `L3 Cache` با اولویت‌بندی `retrieval_optimizer`.
        """
        if time_range:
            start_time = time_range.get("start", datetime.utcnow())
            end_time = time_range.get("end", datetime.utcnow())
        else:
            start_time = datetime.utcnow()
            end_time = datetime.utcnow()

        retrieved_data = await self.db_service.get_time_series_data("conversation_history", user_id, chat_id,
                                                                    start_time, end_time)

        # استخراج پیام‌های فشرده‌شده و باز کردن فشرده‌سازی آن‌ها
        compressed_messages = [entry["compressed_message"] for entry in retrieved_data]
        decompressed_messages = self.quantum_compressor.decompress_messages(
            compressed_messages[0]) if compressed_messages else []

        # استفاده از `retrieval_optimizer` برای بازیابی پیام‌های مهم
        optimized_messages = self.retrieval_optimizer.retrieve_optimized_messages(decompressed_messages)

        return optimized_messages

    async def clear_cache(self, user_id: str, chat_id: str) -> None:
        """
        حذف تمامی مکالمات کاربر از `L3 Cache` (`TimescaleDB`).
        """
        await self.db_service.delete_user_data("conversation_history", user_id, chat_id)

    async def process(self, user_id: str, chat_id: str, messages: List[str]) -> Dict[str, Any]:
        """
        ذخیره و بازیابی داده‌های بلندمدت مکالمه.
        """
        await self.store_messages(user_id, chat_id, messages)
        stored_conversations = await self.retrieve_messages(user_id, chat_id)

        return {
            "user_id": user_id,
            "chat_id": chat_id,
            "conversation_history": stored_conversations,
        }


# تست اولیه ماژول
if __name__ == "__main__":
    import asyncio


    async def test_l3_cache():
        l3_cache = L3Cache()

        user_id = "user_789"
        chat_id = "chat_101"
        messages = [
            "چطور شبکه‌های عصبی عمیق آموزش داده می‌شوند؟",
            "روش‌های بهینه‌سازی مدل‌های زبانی چیست؟",
            "GPT-4 چگونه با مدل‌های قبلی مقایسه می‌شود؟"
        ]

        result = await l3_cache.process(user_id, chat_id, messages)
        print("\n🔹 Updated L3 Cache:")
        print(result)

        print("\n🔹 Retrieving Long-Term Conversations:")
        retrieved_data = await l3_cache.retrieve_messages(user_id, chat_id)
        print(retrieved_data)


    asyncio.run(test_l3_cache())
