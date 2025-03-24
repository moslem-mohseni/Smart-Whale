import asyncio
from typing import Dict, Any, List, Optional
from ai.models.language.core.optimizer.retrieval_optimizer import RetrievalOptimizer
from ai.models.language.core.optimizer.quantum_compressor import QuantumCompressor
from ai.models.language.core.processor.quantum_vectorizer import QuantumVectorizer


class QuantumMemory:
    """
    این کلاس مسئول مدیریت حافظه‌ی کوانتومی برای ذخیره و بازیابی داده‌های پردازشی مهم در مکالمات است.
    """

    def __init__(self, max_entries: int = 1000):
        """
        مقداردهی اولیه `QuantumMemory` با ابزارهای بهینه‌سازی.
        :param max_entries: حداکثر تعداد ورودی‌هایی که در حافظه‌ی کوانتومی ذخیره می‌شود.
        """
        self.max_entries = max_entries
        self.retrieval_optimizer = RetrievalOptimizer()
        self.quantum_compressor = QuantumCompressor()
        self.quantum_vectorizer = QuantumVectorizer()
        self.memory_store: Dict[str, List[Dict[str, Any]]] = {}

    async def store_conversation(self, user_id: str, chat_id: str, messages: List[str]) -> None:
        """
        ذخیره‌ی داده‌های مکالمه‌ای مهم در `Quantum Memory` پس از برداری‌سازی و فشرده‌سازی.
        """
        key = f"quantum_memory:{user_id}:{chat_id}"

        # برداری‌سازی پیام‌ها
        vectorized_data = self.quantum_vectorizer.vectorize_messages(messages)

        # فشرده‌سازی پیام‌ها
        compressed_messages = self.quantum_compressor.compress_messages(messages)

        entry = {
            "vectorized_data": vectorized_data,
            "compressed_messages": compressed_messages
        }

        if key not in self.memory_store:
            self.memory_store[key] = []

        self.memory_store[key].append(entry)

        # اگر تعداد ذخیره‌ها از حد مجاز عبور کند، قدیمی‌ترین داده حذف می‌شود
        if len(self.memory_store[key]) > self.max_entries:
            self.memory_store[key].pop(0)

    async def retrieve_conversation(self, user_id: str, chat_id: str) -> List[str]:
        """
        بازیابی مکالمات ذخیره‌شده در `Quantum Memory` و باز کردن فشرده‌سازی داده‌ها.
        """
        key = f"quantum_memory:{user_id}:{chat_id}"

        if key in self.memory_store:
            latest_entry = self.memory_store[key][-1]
            decompressed_messages = self.quantum_compressor.decompress_messages(latest_entry["compressed_messages"])

            # استفاده از `retrieval_optimizer` برای بازیابی داده‌های کلیدی
            optimized_messages = self.retrieval_optimizer.retrieve_optimized_messages(decompressed_messages)

            return optimized_messages

        return []

    async def clear_memory(self, user_id: str, chat_id: str) -> None:
        """
        حذف تمامی مکالمات کاربر از `Quantum Memory`.
        """
        key = f"quantum_memory:{user_id}:{chat_id}"
        if key in self.memory_store:
            del self.memory_store[key]

    async def process(self, user_id: str, chat_id: str, messages: List[str]) -> Dict[str, Any]:
        """
        ذخیره و بازیابی داده‌های مکالمه‌ای کوانتومی.
        """
        await self.store_conversation(user_id, chat_id, messages)
        stored_conversations = await self.retrieve_conversation(user_id, chat_id)

        return {
            "user_id": user_id,
            "chat_id": chat_id,
            "quantum_memory_data": stored_conversations,
        }


# تست اولیه ماژول
if __name__ == "__main__":
    async def test_quantum_memory():
        quantum_memory = QuantumMemory()

        user_id = "user_999"
        chat_id = "chat_111"
        messages = [
            "چطور مدل‌های یادگیری عمیق بهینه‌سازی می‌شوند؟",
            "تفاوت بین `Batch Normalization` و `Layer Normalization` چیست؟",
            "مدل‌های Transformer چگونه کار می‌کنند؟"
        ]

        result = await quantum_memory.process(user_id, chat_id, messages)
        print("\n🔹 Updated Quantum Memory:")
        print(result)

        print("\n🔹 Retrieving Quantum Memory Conversations:")
        retrieved_data = await quantum_memory.retrieve_conversation(user_id, chat_id)
        print(retrieved_data)


    asyncio.run(test_quantum_memory())
