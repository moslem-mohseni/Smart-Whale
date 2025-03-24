import asyncio
from typing import Dict, Any, List, Optional
from .l1_cache import L1Cache
from .l2_cache import L2Cache
from .l3_cache import L3Cache
from .cache_synchronizer import CacheSynchronizer
from ai.models.language.core.optimizer.retrieval_optimizer import RetrievalOptimizer
from ai.models.language.core.optimizer.quantum_compressor import QuantumCompressor
from ai.models.language.core.optimizer.adaptive_optimizer import AdaptiveOptimizer
from ai.models.language.core.optimizer.quantum_allocator import QuantumAllocator


class MemoryManager:
    """
    مدیریت حافظه‌ی مکالمه‌ای کاربران و بهینه‌سازی ذخیره‌سازی در `L1 Cache`, `L2 Cache`, و `L3 Cache`.
    """

    def __init__(self):
        """
        مقداردهی اولیه‌ی `MemoryManager` با اتصال به لایه‌های کش و بهینه‌سازها.
        """
        self.l1_cache = L1Cache()
        self.l2_cache = L2Cache()
        self.l3_cache = L3Cache()
        self.cache_sync = CacheSynchronizer()
        self.retrieval_optimizer = RetrievalOptimizer()
        self.quantum_compressor = QuantumCompressor()
        self.adaptive_optimizer = AdaptiveOptimizer()
        self.quantum_allocator = QuantumAllocator()

    async def store_message(self, user_id: str, chat_id: str, message: str) -> None:
        """
        ذخیره‌ی پیام جدید در کش‌های مختلف و اعمال بهینه‌سازی‌های لازم.
        """
        # تخصیص منابع پردازشی
        allocation_level = self.quantum_allocator.allocate_processing_level(message)

        # ذخیره‌ی داده در `L1 Cache`
        await self.l1_cache.store_message(user_id, chat_id, message)

        # ذخیره‌ی داده در `L2 Cache`
        await self.l2_cache.store_message(user_id, chat_id, message)

        # بررسی شرایط انتقال به `L3 Cache`
        message_count = await self.l2_cache.get_message_count(user_id, chat_id)
        if message_count >= 10:
            messages = await self.l2_cache.retrieve_messages(user_id, chat_id)
            compressed_messages = self.quantum_compressor.compress_messages(messages)

            await self.l3_cache.store_messages(user_id, chat_id, compressed_messages)
            await self.l2_cache.clear_cache(user_id, chat_id)  # پاک‌سازی `L2 Cache` بعد از انتقال

    async def retrieve_context(self, user_id: str, chat_id: str) -> Dict[str, Any]:
        """
        بازیابی اطلاعات مکالمه از کش‌ها و اعمال بهینه‌سازی‌های بازیابی.
        """
        # بررسی `L1 Cache`
        l1_messages = await self.l1_cache.retrieve_messages(user_id, chat_id)
        if l1_messages:
            return {"messages": l1_messages, "source": "L1 Cache"}

        # بررسی `L2 Cache`
        l2_messages = await self.l2_cache.retrieve_messages(user_id, chat_id)
        if l2_messages:
            return {"messages": l2_messages, "source": "L2 Cache"}

        # بازیابی از `L3 Cache` و استفاده از `retrieval_optimizer`
        l3_messages = await self.retrieval_optimizer.retrieve_optimized_messages(user_id, chat_id)
        if l3_messages:
            await self.l2_cache.batch_store_messages(user_id, chat_id, l3_messages[:10])
            return {"messages": l3_messages, "source": "L3 Cache"}

        return {"messages": [], "source": "No Data"}

    async def clear_memory(self, user_id: str, chat_id: str) -> None:
        """
        پاک‌سازی کامل حافظه‌ی کاربر برای یک چت خاص.
        """
        await self.l1_cache.clear_cache(user_id, chat_id)
        await self.l2_cache.clear_cache(user_id, chat_id)
        await self.l3_cache.clear_cache(user_id, chat_id)

    async def process_message(self, user_id: str, chat_id: str, message: str) -> Dict[str, Any]:
        """
        پردازش پیام جدید، ذخیره در کش‌ها و مدیریت بازیابی.
        """
        await self.store_message(user_id, chat_id, message)
        return await self.retrieve_context(user_id, chat_id)


# تست اولیه ماژول
if __name__ == "__main__":
    async def test_memory_manager():
        memory_manager = MemoryManager()
        user_id = "user_123"
        chat_id = "chat_456"

        messages = [
            "سلام، امروز چه خبر؟",
            "می‌توانی درباره‌ی یادگیری ماشین توضیح بدهی؟",
            "بهترین مدل‌های پردازش زبان کدامند؟",
            "چگونه یک مدل یادگیری عمیق آموزش دهیم؟",
            "آیا GPT-4 بهتر از BERT است؟"
        ]

        for msg in messages:
            result = await memory_manager.process_message(user_id, chat_id, msg)
            print("\n🔹 Updated Memory State:")
            print(result)


    asyncio.run(test_memory_manager())
