import asyncio
from typing import List, Tuple, Optional
from infrastructure.vector_store import VectorStore
import numpy as np

class VectorSearch:
    """
    این کلاس مسئول پردازش برداری پیام‌های مکالمه‌ای و یافتن شباهت‌ها بین آن‌ها است.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        """
        مقداردهی اولیه `VectorSearch` با اتصال به دیتابیس برداری.
        :param similarity_threshold: حد آستانه برای تشخیص شباهت موضوعات (بین `0` و `1`، پیش‌فرض `0.75`)
        """
        self.vector_store = VectorStore()
        self.similarity_threshold = similarity_threshold

    async def embed_text(self, text: str) -> List[float]:
        """
        تبدیل متن به بردار ویژگی برای استفاده در جستجوی برداری.
        """
        return await self.vector_store.generate_embedding(text)

    async def store_vector(self, text: str, category: Optional[str] = None) -> None:
        """
        ذخیره‌ی بردار متن در پایگاه داده‌ی برداری.
        """
        embedding = await self.embed_text(text)
        await self.vector_store.store_vector(embedding, text, category)

    async def find_similar(self, query: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        یافتن پیام‌های مشابه بر اساس بردار متنی و جستجوی برداری.
        """
        query_vector = await self.embed_text(query)
        stored_vectors = await self.vector_store.get_all_vectors()

        # محاسبه‌ی شباهت کسینوسی بین بردارهای ذخیره‌شده و کوئری
        similarities = []
        for stored_text, stored_vector in stored_vectors.items():
            similarity = self.cosine_similarity(query_vector, stored_vector)
            if similarity >= self.similarity_threshold:
                similarities.append((stored_text, similarity))

        # مرتب‌سازی بر اساس بالاترین میزان شباهت
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        محاسبه‌ی شباهت کسینوسی بین دو بردار.
        """
        vec1, vec2 = np.array(vec1), np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 * norm_vec2 != 0 else 0.0

# تست اولیه ماژول
if __name__ == "__main__":
    async def test_vector_search():
        vector_search = VectorSearch()

        await vector_search.store_vector("یادگیری ماشین چیست؟")
        await vector_search.store_vector("یادگیری عمیق و شبکه‌های عصبی")
        await vector_search.store_vector("مدل‌های GPT-3 و BERT")

        query = "یادگیری شبکه عصبی"
        results = await vector_search.find_similar(query)

        print("\n🔹 Similar Topics Found:")
        for text, similarity in results:
            print(f"🔹 {text} - شباهت: {similarity:.2f}")

    asyncio.run(test_vector_search())
