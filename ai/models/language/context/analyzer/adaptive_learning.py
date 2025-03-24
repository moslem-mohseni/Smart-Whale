import asyncio
from typing import Dict, Any, List, Optional
from ai.models.language.context.analyzer.topic_store import TopicStore
from ai.models.language.context.analyzer.semantic_analyzer import SemanticAnalyzer

class AdaptiveLearning:
    """
    این کلاس مسئول یادگیری تطبیقی و به‌روزرسانی موضوعات جدید در پایگاه داده‌ی موضوعات است.
    """

    def __init__(self, confidence_threshold: float = 0.75):
        """
        مقداردهی اولیه `AdaptiveLearning` با ابزارهای پردازشی.
        :param confidence_threshold: حد آستانه‌ی اطمینان برای اضافه کردن موضوع جدید.
        """
        self.topic_store = TopicStore()
        self.semantic_analyzer = SemanticAnalyzer()
        self.confidence_threshold = confidence_threshold

    async def process_new_message(self, message: str) -> Dict[str, Any]:
        """
        پردازش پیام جدید برای تشخیص و ذخیره‌ی موضوع جدید در صورت نیاز.
        """
        analysis_result = await self.semantic_analyzer.analyze_text(message)
        predicted_topic = analysis_result["predicted_topic"]
        confidence_score = analysis_result["confidence_score"]

        # اگر موضوع در پایگاه داده نباشد و میزان اطمینان بالا باشد، آن را ذخیره می‌کنیم.
        existing_topics = await self.topic_store.get_all_topics()
        if predicted_topic not in existing_topics and confidence_score >= self.confidence_threshold:
            await self.topic_store.add_new_topic(predicted_topic)
            is_new_topic = True
        else:
            is_new_topic = False

        return {
            "input_text": message,
            "detected_topic": predicted_topic,
            "confidence_score": confidence_score,
            "new_topic_added": is_new_topic
        }

# تست اولیه ماژول
if __name__ == "__main__":
    async def test_adaptive_learning():
        adaptive_learning = AdaptiveLearning()

        messages = [
            "مدل‌های Transformer چگونه کار می‌کنند؟",
            "یادگیری ماشین و شبکه‌های عصبی چه تفاوتی دارند؟",
            "تفاوت بین CNN و RNN چیست؟"
        ]

        for msg in messages:
            result = await adaptive_learning.process_new_message(msg)
            print("\n🔹 Adaptive Learning Result:")
            print(result)

    asyncio.run(test_adaptive_learning())
