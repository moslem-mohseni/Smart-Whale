import asyncio
from typing import List, Dict, Optional
from ai.models.language.infrastructure.vector_store import VectorStore

class TopicStore:
    """
    این کلاس مسئول ذخیره‌سازی و مدیریت پایگاه داده‌ی برداری موضوعات است.
    """

    def __init__(self):
        """
        مقداردهی اولیه `TopicStore` با اتصال به دیتابیس برداری.
        """
        self.vector_store = VectorStore()

    async def add_new_topic(self, topic: str) -> None:
        """
        اضافه کردن یک موضوع جدید به پایگاه داده‌ی برداری.
        """
        embedding = await self.vector_store.generate_embedding(topic)
        await self.vector_store.store_vector(embedding, topic, category="topic")

    async def get_all_topics(self) -> List[str]:
        """
        بازیابی همه‌ی موضوعات ذخیره‌شده.
        """
        stored_vectors = await self.vector_store.get_all_vectors()
        return list(stored_vectors.keys())

    async def find_closest_topic(self, query: str) -> Optional[str]:
        """
        یافتن نزدیک‌ترین موضوع به یک ورودی جدید.
        """
        query_embedding = await self.vector_store.generate_embedding(query)
        stored_vectors = await self.vector_store.get_all_vectors()

        max_similarity = 0
        closest_topic = None

        for topic, stored_embedding in stored_vectors.items():
            similarity = self.vector_store.calculate_similarity(query_embedding, stored_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                closest_topic = topic

        return closest_topic if max_similarity >= 0.7 else None  # حد آستانه‌ی اطمینان ۰.۷

# تست اولیه ماژول
if __name__ == "__main__":
    async def test_topic_store():
        topic_store = TopicStore()

        await topic_store.add_new_topic("پردازش زبان طبیعی")
        await topic_store.add_new_topic("یادگیری عمیق")

        all_topics = await topic_store.get_all_topics()
        print("\n🔹 Stored Topics:")
        print(all_topics)

        query = "NLP چیست؟"
        closest_topic = await topic_store.find_closest_topic(query)
        print("\n🔹 Closest Topic Found:")
        print(closest_topic)

    asyncio.run(test_topic_store())
