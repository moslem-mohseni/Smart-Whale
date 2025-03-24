import asyncio
from typing import Dict, Any, List, Optional
from .l1_cache import L1Cache
from .l2_cache import L2Cache
from .l3_cache import L3Cache
from ai.models.language.core.optimizer.retrieval_optimizer import RetrievalOptimizer
from ai.models.language.core.optimizer.quantum_compressor import QuantumCompressor
from ai.models.language.core.optimizer.load_balancer import LoadBalancer

class CacheSynchronizer:
    """
    این کلاس مسئول هماهنگ‌سازی بین `L1 Cache`, `L2 Cache`, و `L3 Cache` است.
    داده‌های جدید را به `Batch` به `L3 Cache` منتقل کرده و در صورت نیاز `L2 Cache` را مقداردهی می‌کند.
    """

    def __init__(self):
        """
        مقداردهی اولیه `CacheSynchronizer` و اتصال به لایه‌های کش.
        """
        self.l1_cache = L1Cache()
        self.l2_cache = L2Cache()
        self.l3_cache = L3Cache()
        self.retrieval_optimizer = RetrievalOptimizer()
        self.quantum_compressor = QuantumCompressor()
        self.load_balancer = LoadBalancer()

    async def sync_to_l3_cache(self, user_id: str, chat_id: str) -> None:
        """
        انتقال داده‌ها از `L2 Cache` به `L3 Cache` در `Batch`.
        """
        messages = await self.l2_cache.retrieve_messages(user_id, chat_id)
        if messages:
            compressed_messages = self.quantum_compressor.compress_messages([msg["text"] for msg in messages])
            await self.l3_cache.store_messages(user_id, chat_id, compressed_messages)
            await self.l2_cache.clear_cache(user_id, chat_id)  # پاک‌سازی `L2 Cache` بعد از انتقال

    async def retrieve_from_l3_to_l2(self, user_id: str, chat_id: str) -> None:
        """
        بازیابی داده‌ها از `L3 Cache` و مقداردهی `L2 Cache` برای پاسخ‌دهی سریع‌تر.
        """
        messages = await self.l3_cache.retrieve_messages(user_id, chat_id)
        if messages:
            await self.l2_cache.batch_store_messages(user_id, chat_id, messages[:10])  # ذخیره‌ی ۱۰ پیام آخر

    async def clean_l1_cache(self, user_id: str, chat_id: str) -> None:
        """
        پاک‌سازی `L1 Cache` در صورت وجود داده‌های قدیمی یا کم‌اهمیت.
        """
        await self.l1_cache.clear_cache(user_id, chat_id)

    async def sync_and_retrieve(self, user_id: str, chat_id: str) -> Dict[str, Any]:
        """
        همگام‌سازی کش و بازیابی داده‌های مکالمه.
        """
        # بررسی `L1 Cache`
        l1_messages = await self.l1_cache.retrieve_messages(user_id, chat_id)
        if l1_messages:
            return {"messages": l1_messages, "source": "L1 Cache"}

        # بررسی `L2 Cache`
        l2_messages = await self.l2_cache.retrieve_messages(user_id, chat_id)
        if l2_messages:
            return {"messages": l2_messages, "source": "L2 Cache"}

        # بازیابی از `L3 Cache` و مقداردهی `L2 Cache`
        await self.retrieve_from_l3_to_l2(user_id, chat_id)
        l3_messages = await self.retrieval_optimizer.retrieve_optimized_messages(user_id, chat_id)

        return {"messages": l3_messages, "source": "L3 Cache" if l3_messages else "No Data"}

# تست اولیه ماژول
if __name__ == "__main__":
    async def test_cache_synchronizer():
        cache_sync = CacheSynchronizer()

        user_id = "user_123"
        chat_id = "chat_456"

        # انتقال `L2 Cache` به `L3 Cache`
        await cache_sync.sync_to_l3_cache(user_id, chat_id)

        # بازیابی از `L3 Cache` و مقداردهی `L2 Cache`
        await cache_sync.retrieve_from_l3_to_l2(user_id, chat_id)

        # پاک‌سازی `L1 Cache`
        await cache_sync.clean_l1_cache(user_id, chat_id)

        # بازیابی نهایی از مناسب‌ترین لایه‌ی کش
        result = await cache_sync.sync_and_retrieve(user_id, chat_id)
        print("\n🔹 Final Cache Retrieval:")
        print(result)

    asyncio.run(test_cache_synchronizer())
