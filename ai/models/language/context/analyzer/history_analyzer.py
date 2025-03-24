import asyncio
from typing import Dict, Any, List, Optional
from ai.models.language.context.memory.l3_cache import L3Cache
from ai.models.language.context.analyzer.context_detector import ContextDetector
from ai.models.language.context.analyzer.relevance_checker import RelevanceChecker
from ai.models.language.context.analyzer.summarizer import Summarizer

class HistoryAnalyzer:
    """
    این کلاس مسئول تحلیل تاریخچه‌ی مکالمات کاربران برای درک روند گفتگو و تشخیص تغییرات زمینه‌ای است.
    """

    def __init__(self, analysis_window: int = 20):
        """
        مقداردهی اولیه `HistoryAnalyzer` با ابزارهای پردازشی.
        :param analysis_window: تعداد پیام‌های گذشته که باید تحلیل شوند.
        """
        self.l3_cache = L3Cache()
        self.context_detector = ContextDetector()
        self.relevance_checker = RelevanceChecker()
        self.summarizer = Summarizer()
        self.analysis_window = analysis_window

    async def analyze_history(self, user_id: str, chat_id: str) -> Dict[str, Any]:
        """
        بررسی پیام‌های گذشته برای تحلیل زمینه‌ی مکالمه و تشخیص الگوهای رفتاری.
        """
        messages = await self.l3_cache.retrieve_messages(user_id, chat_id)

        if not messages:
            return {"history_context": "No previous data found", "patterns": []}

        # دریافت جدیدترین پیام‌های ذخیره‌شده
        latest_messages = messages[-self.analysis_window:]

        # بررسی تغییرات زمینه‌ای در مکالمه
        context_changes = [await self.context_detector.detect_context(user_id, chat_id, msg) for msg in latest_messages]

        # تحلیل میزان تکرار موضوعات
        frequent_topics = self.find_frequent_topics(latest_messages)

        # تولید خلاصه‌ای از مکالمات طولانی برای بهینه‌سازی ذخیره‌سازی
        summary_result = await self.summarizer.generate_summary(user_id, chat_id)

        return {
            "history_context": latest_messages,
            "context_changes": context_changes,
            "frequent_topics": frequent_topics,
            "conversation_summary": summary_result["summary"]
        }

    def find_frequent_topics(self, messages: List[str]) -> Dict[str, int]:
        """
        تحلیل تعداد تکرار موضوعات در مکالمات گذشته.
        """
        topic_count = {}
        for message in messages:
            topic = self.extract_topic(message)
            if topic:
                topic_count[topic] = topic_count.get(topic, 0) + 1

        return topic_count

    async def extract_topic(self, message: str) -> Optional[str]:
        """
        استخراج موضوع اصلی از پیام با تحلیل برداری و معنایی.
        """
        context_result = await self.context_detector.detect_context("", "", message)
        return context_result["context"] if not context_result["context_changed"] else None

# تست اولیه ماژول
if __name__ == "__main__":
    async def test_history_analyzer():
        history_analyzer = HistoryAnalyzer()

        user_id = "user_456"
        chat_id = "chat_789"

        result = await history_analyzer.analyze_history(user_id, chat_id)
        print("\n🔹 History Analysis Result:")
        print(result)

    asyncio.run(test_history_analyzer())
