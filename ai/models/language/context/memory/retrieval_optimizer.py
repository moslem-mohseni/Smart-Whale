import asyncio
from typing import Dict, Any, List
from ai.models.language.core.processor.quantum_vectorizer import QuantumVectorizer
from ai.models.language.core.processor.feature_extractor import FeatureExtractor

class RetrievalOptimizer:
    """
    این کلاس مسئول بهینه‌سازی بازیابی داده‌های مکالمه‌ای است.
    از روش‌های مختلف جستجو (`Keyword`, `Semantic`, `Hybrid`) برای یافتن داده‌های کلیدی استفاده می‌کند.
    """

    def __init__(self):
        """
        مقداردهی اولیه `RetrievalOptimizer` با ابزارهای پردازشی.
        """
        self.quantum_vectorizer = QuantumVectorizer()
        self.feature_extractor = FeatureExtractor()

    def keyword_search(self, query: str, messages: List[str]) -> List[str]:
        """
        جستجوی کلیدواژه‌ای برای یافتن داده‌های مکالمه‌ای مرتبط.
        """
        return [msg for msg in messages if query.lower() in msg.lower()]

    def semantic_search(self, query: str, messages: List[str]) -> List[str]:
        """
        جستجوی معنایی برای یافتن داده‌های مشابه از لحاظ مفهومی.
        """
        query_vector = self.quantum_vectorizer.vectorize_text(query)
        message_vectors = {msg: self.quantum_vectorizer.vectorize_text(msg) for msg in messages}

        # محاسبه‌ی شباهت و بازگرداندن پیام‌های مشابه
        sorted_messages = sorted(messages, key=lambda msg: self.quantum_vectorizer.cosine_similarity(query_vector, message_vectors[msg]), reverse=True)
        return sorted_messages[:5]  # بازگرداندن ۵ پیام مرتبط‌تر

    def hybrid_search(self, query: str, messages: List[str]) -> List[str]:
        """
        جستجوی ترکیبی که از `Keyword` و `Semantic` به‌صورت همزمان استفاده می‌کند.
        """
        keyword_results = set(self.keyword_search(query, messages))
        semantic_results = set(self.semantic_search(query, messages))

        return list(keyword_results.union(semantic_results))  # ترکیب نتایج دو روش

    def retrieve_optimized_messages(self, messages: List[str], query: str = None) -> List[str]:
        """
        انتخاب هوشمندانه‌ی پیام‌های مهم از مکالمه‌ها برای افزایش سرعت پردازش.
        """
        if not messages:
            return []

        if query:
            return self.hybrid_search(query, messages)

        # اگر هیچ `Query` مشخص نشده باشد، مهم‌ترین پیام‌ها را بر اساس `FeatureExtractor` انتخاب کن
        important_messages = sorted(messages, key=lambda msg: self.feature_extractor.extract_relevance_score(msg), reverse=True)
        return important_messages[:10]  # بازگرداندن ۱۰ پیام مهم‌تر

# تست اولیه ماژول
if __name__ == "__main__":
    async def test_retrieval_optimizer():
        retrieval_optimizer = RetrievalOptimizer()

        messages = [
            "سلام، می‌توانی درباره هوش مصنوعی توضیح بدهی؟",
            "یادگیری ماشین چیست و چه تفاوتی با یادگیری عمیق دارد؟",
            "مدل‌های Transformer چگونه کار می‌کنند؟",
            "چه تفاوتی بین `Batch Normalization` و `Layer Normalization` وجود دارد؟",
            "بهترین روش‌های بهینه‌سازی مدل‌های یادگیری عمیق چیست؟"
        ]

        query = "یادگیری ماشین"

        result = retrieval_optimizer.retrieve_optimized_messages(messages, query)
        print("\n🔹 Optimized Messages Retrieval:")
        print(result)

    asyncio.run(test_retrieval_optimizer())
