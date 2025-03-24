import asyncio
from typing import Dict, Any, List, Tuple
from ai.models.language.context.analyzer.vector_search import VectorSearch
from ai.models.language.context.analyzer.context_detector import ContextDetector
from ai.models.language.context.analyzer.history_analyzer import HistoryAnalyzer

class RelevanceChecker:
    """
    این کلاس مسئول بررسی میزان ارتباط پیام جدید با مکالمات قبلی است.
    """

    def __init__(self, relevance_threshold: float = 0.7):
        """
        مقداردهی اولیه `RelevanceChecker` با ابزارهای پردازشی.
        :param relevance_threshold: حد آستانه‌ی ارتباط برای تشخیص میزان همخوانی پیام جدید با مکالمه‌ی قبلی.
        """
        self.vector_search = VectorSearch()
        self.context_detector = ContextDetector()
        self.history_analyzer = HistoryAnalyzer()
        self.relevance_threshold = relevance_threshold

    async def check_relevance(self, user_id: str, chat_id: str, message: str) -> Dict[str, Any]:
        """
        بررسی ارتباط پیام جدید با مکالمات قبلی.
        """
        # دریافت مکالمات قبلی از `history_analyzer.py`
        history_data = await self.history_analyzer.analyze_history(user_id, chat_id)
        previous_messages = history_data["history_context"]

        if not previous_messages:
            return {
                "relevance_score": 0.0,
                "is_relevant": False,
                "reason": "No previous conversation history found."
            }

        # بررسی شباهت پیام جدید با پیام‌های قبلی
        similarities = await self.vector_search.find_similar(message, top_n=5)
        max_similarity = max([sim[1] for sim in similarities]) if similarities else 0.0

        # تشخیص تغییر زمینه از `context_detector.py`
        context_result = await self.context_detector.detect_context(user_id, chat_id, message)
        context_changed = context_result["context_changed"]

        is_relevant = max_similarity >= self.relevance_threshold and not context_changed

        return {
            "relevance_score": max_similarity,
            "is_relevant": is_relevant,
            "context_changed": context_changed,
            "reason": "Relevant to previous conversation" if is_relevant else "Not relevant or new topic detected."
        }

# تست اولیه ماژول
if __name__ == "__main__":
    async def test_relevance_checker():
        relevance_checker = RelevanceChecker()

        user_id = "user_123"
        chat_id = "chat_456"
        test_messages = [
            "می‌توانی درباره شبکه‌های عصبی توضیح بدهی؟",
            "تفاوت بین CNN و RNN چیست؟",
            "بهترین روش برای تنظیم `learning rate` چیست؟"
        ]

        for msg in test_messages:
            result = await relevance_checker.check_relevance(user_id, chat_id, msg)
            print("\n🔹 Relevance Check Result:")
            print(result)

    asyncio.run(test_relevance_checker())
