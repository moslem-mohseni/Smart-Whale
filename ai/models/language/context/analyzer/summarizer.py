import asyncio
from typing import Dict, Any, List
from ai.models.language.context.analyzer.history_analyzer import HistoryAnalyzer
from ai.models.language.context.analyzer.semantic_analyzer import SemanticAnalyzer
from ai.models.language.context.memory.l3_cache import L3Cache

class Summarizer:
    """
    این کلاس مسئول خلاصه‌سازی مکالمات طولانی برای بهینه‌سازی پردازش و ذخیره‌سازی است.
    """

    def __init__(self, summary_length: int = 5):
        """
        مقداردهی اولیه `Summarizer` با ابزارهای پردازشی.
        :param summary_length: تعداد جملات خلاصه‌شده‌ی نهایی
        """
        self.history_analyzer = HistoryAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.l3_cache = L3Cache()
        self.summary_length = summary_length

    async def generate_summary(self, user_id: str, chat_id: str) -> Dict[str, Any]:
        """
        تولید خلاصه‌ای از مکالمات اخیر برای کاهش حجم داده‌های ذخیره‌شده.
        """
        # دریافت مکالمات قبلی از `history_analyzer.py`
        history_data = await self.history_analyzer.analyze_history(user_id, chat_id)
        previous_messages = history_data["history_context"]

        if not previous_messages:
            return {"summary": "No previous conversation history found."}

        # استخراج جملات کلیدی با تحلیل معنایی
        ranked_messages = sorted(
            previous_messages,
            key=lambda msg: self.semantic_analyzer.analyze_text(msg)["confidence_score"],
            reverse=True
        )

        # انتخاب مهم‌ترین جملات برای خلاصه‌سازی
        summary = ranked_messages[:self.summary_length]

        # ذخیره‌ی خلاصه‌ی تولید‌شده در `L3 Cache`
        await self.l3_cache.store_messages(user_id, chat_id, summary)

        return {"summary": summary}

# تست اولیه ماژول
if __name__ == "__main__":
    async def test_summarizer():
        summarizer = Summarizer()

        user_id = "user_123"
        chat_id = "chat_456"

        result = await summarizer.generate_summary(user_id, chat_id)
        print("\n🔹 Summary Generation Result:")
        print(result)

    asyncio.run(test_summarizer())
