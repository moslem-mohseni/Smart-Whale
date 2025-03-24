import asyncio
from typing import Dict, Any
from ai.models.language.context.analyzer.vector_search import VectorSearch
from ai.models.language.context.analyzer.semantic_analyzer import SemanticAnalyzer
from ai.models.language.context.analyzer.history_analyzer import HistoryAnalyzer
from ai.models.language.context.analyzer.relevance_checker import RelevanceChecker
from ai.models.language.context.memory.l2_cache import L2Cache
from ai.models.language.context.memory.l3_cache import L3Cache

class ContextDetector:
    """
    این کلاس مسئول تشخیص زمینه‌ی مکالمه و تغییرات آن در طول گفتگو است.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        """
        مقداردهی اولیه `ContextDetector` با ابزارهای پردازشی.
        :param similarity_threshold: حد آستانه برای تشخیص تغییر زمینه‌ی مکالمه
        """
        self.vector_search = VectorSearch()
        self.semantic_analyzer = SemanticAnalyzer()
        self.history_analyzer = HistoryAnalyzer()
        self.relevance_checker = RelevanceChecker()
        self.l2_cache = L2Cache()
        self.l3_cache = L3Cache()
        self.similarity_threshold = similarity_threshold

    async def detect_context(self, user_id: str, chat_id: str, message: str) -> Dict[str, Any]:
        """
        تحلیل پیام جدید و تشخیص زمینه‌ی اصلی مکالمه.
        """
        # دریافت مکالمات قبلی از `history_analyzer.py`
        history_data = await self.history_analyzer.analyze_history(user_id, chat_id)
        previous_messages = history_data["history_context"]

        # بررسی میزان ارتباط پیام جدید با مکالمات قبلی
        relevance_result = await self.relevance_checker.check_relevance(user_id, chat_id, message)
        is_relevant = relevance_result["is_relevant"]

        # جستجوی برداری برای تشخیص زمینه
        similar_topics = await self.vector_search.find_similar(message, top_n=3)
        topic_matched = similar_topics[0][1] >= self.similarity_threshold if similar_topics else False

        context_changed = not is_relevant or not topic_matched

        return {
            "context": similar_topics[0][0] if topic_matched else "New Context Detected",
            "context_changed": context_changed
        }

    async def update_context(self, user_id: str, chat_id: str, message: str) -> None:
        """
        به‌روزرسانی زمینه‌ی مکالمه در `L2 Cache` و `L3 Cache` در صورت تغییر.
        """
        detection_result = await self.detect_context(user_id, chat_id, message)

        if detection_result["context_changed"]:
            await self.l2_cache.store_message(user_id, chat_id, {"text": message, "context": "New Context"})
            await self.l3_cache.store_messages(user_id, chat_id, [message])

# تست اولیه ماژول
if __name__ == "__main__":
    async def test_context_detector():
        context_detector = ContextDetector()

        user_id = "user_123"
        chat_id = "chat_789"
        messages = [
            "امروز می‌خواهم درباره یادگیری عمیق صحبت کنم.",
            "مدل‌های یادگیری عمیق چطور کار می‌کنند؟",
            "بیایید درباره الگوریتم‌های بهینه‌سازی صحبت کنیم."
        ]

        for msg in messages:
            result = await context_detector.detect_context(user_id, chat_id, msg)
            print("\n🔹 Context Detection Result:")
            print(result)

            await context_detector.update_context(user_id, chat_id, msg)

    asyncio.run(test_context_detector())
