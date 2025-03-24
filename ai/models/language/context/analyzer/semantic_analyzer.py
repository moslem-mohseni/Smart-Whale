import asyncio
from typing import List, Dict, Any, Optional
from ai.models.language.context.analyzer.vector_search import VectorSearch
from ai.models.language.context.retriever.topic_store import TopicStore
from ai.models.language.core.processor.language_model_interface import LanguageModelInterface

class SemanticAnalyzer:
    """
    این کلاس مسئول تحلیل معنایی پیام‌های کاربران و استخراج مفاهیم کلیدی از آن‌ها است.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """
        مقداردهی اولیه `SemanticAnalyzer` با ابزارهای پردازشی.
        :param confidence_threshold: حد آستانه‌ی اطمینان برای تشخیص موضوع
        """
        self.vector_search = VectorSearch()
        self.topic_store = TopicStore()
        self.language_model = LanguageModelInterface()
        self.confidence_threshold = confidence_threshold

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        تحلیل معنایی متن و استخراج مفهوم کلیدی.
        """
        # تبدیل متن به نمایش برداری
        embedding = await self.vector_search.embed_text(text)

        # بررسی پایگاه داده‌ی موضوعات برای یافتن نزدیک‌ترین دسته
        similar_topics = await self.vector_search.find_similar(text, top_n=3)

        # اگر موضوع مشابهی پیدا نشد، مدل زبانی موضوع جدید را پیشنهاد می‌دهد
        if not similar_topics or similar_topics[0][1] < self.confidence_threshold:
            predicted_topic = await self.language_model.predict_topic(text)
            await self.topic_store.add_new_topic(predicted_topic)
        else:
            predicted_topic = similar_topics[0][0]

        return {
            "input_text": text,
            "predicted_topic": predicted_topic,
            "confidence_score": similar_topics[0][1] if similar_topics else 0
        }

# تست اولیه ماژول
if __name__ == "__main__":
    async def test_semantic_analyzer():
        semantic_analyzer = SemanticAnalyzer()

        messages = [
            "چطور مدل‌های یادگیری عمیق کار می‌کنند؟",
            "تفاوت بین `Batch Normalization` و `Layer Normalization` چیست؟",
            "BERT و GPT چه تفاوتی دارند؟"
        ]

        for msg in messages:
            result = await semantic_analyzer.analyze_text(msg)
            print("\n🔹 Semantic Analysis Result:")
            print(result)

    asyncio.run(test_semantic_analyzer())
