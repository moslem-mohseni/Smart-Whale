# persian/language_processors/contextual/evaluator.py
"""
ماژول evaluator.py

این ماژول شامل کلاس Evaluator است که وظیفه‌ی ارزیابی کیفیت زمینه (Context) در پردازش زبان فارسی را بر عهده دارد.
این ارزیابی شامل محاسبه‌ی امتیاز کیفیت کلی زمینه، رتبه‌بندی پیام‌های مرتبط بر اساس معیارهای تعریف‌شده و تولید گزارش‌های جامع از ارزیابی می‌باشد.

معیارهای ارزیابی ممکن است شامل تعداد پیام‌های مرتبط، میانگین اهمیت پیام‌ها، میزان همبستگی (similarity) میان پیام‌ها، و سایر شاخص‌های استخراج‌شده از داده‌های زمینه باشد.
"""

import time
import logging
import json
import statistics
from typing import List, Dict, Any, Optional

# در صورت نیاز از توابع کمکی موجود در ماژول‌های utils استفاده می‌کنیم
from .config import get_importance_levels, get_confidence_levels
from ..utils.regex_utils import cleanup_text_for_compare
from ..utils.misc import compute_statistics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Evaluator:
    """
    کلاس Evaluator وظیفه ارزیابی کیفیت زمینه و رتبه‌بندی پیام‌های مرتبط با پیام ورودی را بر عهده دارد.

    متدهای اصلی:
    - evaluate_context_quality(context_data): محاسبه امتیاز کلی کیفیت زمینه.
    - rank_context_messages(context_messages): رتبه‌بندی پیام‌های زمینه بر اساس اهمیت و شباهت.
    - generate_context_report(context_data): تولید گزارش جامع از ارزیابی زمینه.
    """

    def __init__(self, weight_importance: float = 0.4, weight_similarity: float = 0.4, weight_topic: float = 0.2):
        """
        مقداردهی اولیه Evaluator.

        Args:
            weight_importance (float): وزن اهمیت پیام‌ها (پیش‌فرض: 0.4).
            weight_similarity (float): وزن شباهت پیام‌ها (پیش‌فرض: 0.4).
            weight_topic (float): وزن مطابقت موضوعات (پیش‌فرض: 0.2).
        """
        self.weight_importance = weight_importance
        self.weight_similarity = weight_similarity
        self.weight_topic = weight_topic
        logger.info("Evaluator با تنظیمات اولیه راه‌اندازی شد.")

    def evaluate_context_quality(self, context_data: Dict[str, Any]) -> float:
        """
        محاسبه کیفیت کلی زمینه بر اساس داده‌های ورودی.

        این متد از پیام‌های زمینه، نیت کاربر، دانش و موجودیت‌های مرتبط استفاده می‌کند تا یک امتیاز کلی بین 0 و 1 ارائه دهد.
        امتیاز کیفیت از ترکیب میانگین اهمیت پیام‌ها، شاخص‌های شباهت و تطابق موضوعات محاسبه می‌شود.

        Args:
            context_data (Dict[str, Any]): دیکشنری شامل اطلاعات زمینه‌ای شامل:
                - conversation_info: اطلاعات کلی مکالمه.
                - context_messages: لیستی از پیام‌های انتخاب‌شده برای زمینه.
                - user_intent: اطلاعات نیت کاربر.
                - related_knowledge: لیستی از دانش‌های استخراج‌شده.
                - related_entities: لیستی از موجودیت‌های مرتبط.

        Returns:
            float: امتیاز کیفیت زمینه (بین 0 و 1).
        """
        try:
            messages = context_data.get('context_messages', [])
            if not messages:
                logger.warning("هیچ پیام زمینه‌ای برای ارزیابی موجود نیست.")
                return 0.0

            # استخراج اهمیت پیام‌ها
            importance_scores = [msg.get('importance', 0.5) for msg in messages]
            importance_avg = statistics.mean(importance_scores) if importance_scores else 0.0

            # محاسبه شاخص شباهت کلی پیام‌ها (با استفاده از توابع کمکی موجود)
            # فرض می‌کنیم که هر پیام دارای فیلد 'similarity' است که توسط تابع رتبه‌بندی محاسبه می‌شود
            similarities = [msg.get('similarity', 0.5) for msg in messages]
            similarity_avg = statistics.mean(similarities) if similarities else 0.0

            # تطابق موضوعات: بررسی تعداد موضوعات مشترک بین نیت کاربر و موضوعات پیام‌ها
            user_intent = context_data.get('user_intent', {})
            intent_topics = set(user_intent.get('topics', []))
            # فرض کنید هر پیام زمینه دارای لیستی از 'topics' است
            message_topics = []
            for msg in messages:
                topics = msg.get('topics', [])
                message_topics.extend(topics)
            message_topics = set(message_topics)
            topic_match_ratio = len(intent_topics.intersection(message_topics)) / (len(intent_topics) or 1)

            # ترکیب امتیازها با استفاده از وزن‌های تعیین‌شده
            quality_score = (
                self.weight_importance * importance_avg +
                self.weight_similarity * similarity_avg +
                self.weight_topic * topic_match_ratio
            )

            # اطمینان از اینکه امتیاز بین 0 و 1 قرار گیرد
            quality_score = max(0.0, min(quality_score, 1.0))
            return round(quality_score, 4)
        except Exception as e:
            logger.error(f"خطا در ارزیابی کیفیت زمینه: {e}")
            return 0.0

    def rank_context_messages(self, context_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        رتبه‌بندی پیام‌های زمینه بر اساس اهمیت و شباهت به پیام ورودی.

        این متد پیام‌های زمینه را بر اساس دو معیار 'importance' و 'similarity' مرتب کرده و پیام‌های پرارزش‌تر را در صدر قرار می‌دهد.

        Args:
            context_messages (List[Dict[str, Any]]): لیستی از پیام‌های زمینه.

        Returns:
            List[Dict[str, Any]]: لیست مرتب‌شده از پیام‌های زمینه.
        """
        try:
            # برای هر پیام امتیاز نهایی = اهمیت * 0.6 + شباهت * 0.4
            for msg in context_messages:
                imp = msg.get('importance', 0.5)
                sim = msg.get('similarity', 0.5)
                msg['final_score'] = 0.6 * imp + 0.4 * sim

            sorted_messages = sorted(context_messages, key=lambda x: x['final_score'], reverse=True)
            return sorted_messages
        except Exception as e:
            logger.error(f"خطا در رتبه‌بندی پیام‌های زمینه: {e}")
            return context_messages

    def generate_context_report(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        تولید گزارش جامع از ارزیابی زمینه.

        این گزارش شامل جزئیات آماری پیام‌های زمینه، نیت کاربر، دانش و موجودیت‌های مرتبط و همچنین امتیاز کیفیت کلی است.

        Args:
            context_data (Dict[str, Any]): داده‌های زمینه استخراج‌شده.

        Returns:
            Dict[str, Any]: گزارش ارزیابی زمینه شامل:
                - quality_score: امتیاز کلی کیفیت زمینه.
                - message_stats: آمار مربوط به پیام‌ها (مانند میانگین اهمیت و شباهت).
                - topic_match_ratio: نسبت تطابق موضوعات.
                - raw_data: داده‌های اولیه زمینه.
        """
        try:
            messages = context_data.get('context_messages', [])
            importance_scores = [msg.get('importance', 0.5) for msg in messages]
            similarity_scores = [msg.get('similarity', 0.5) for msg in messages]

            stats = {
                "importance_mean": round(statistics.mean(importance_scores), 4) if importance_scores else None,
                "similarity_mean": round(statistics.mean(similarity_scores), 4) if similarity_scores else None,
                "message_count": len(messages)
            }

            quality_score = self.evaluate_context_quality(context_data)

            report = {
                "quality_score": quality_score,
                "message_stats": stats,
                "raw_data": context_data
            }
            return report
        except Exception as e:
            logger.error(f"خطا در تولید گزارش زمینه: {e}")
            return {"quality_score": 0.0, "message_stats": {}, "raw_data": context_data}


if __name__ == "__main__":
    # تست نمونه از Evaluator
    sample_context = {
        "conversation_info": {
            "title": "مکالمه تست",
            "context_type": "CHAT",
            "start_time": "2023-03-15T10:00:00",
            "message_count": 5
        },
        "context_messages": [
            {"content": "سلام، امروز هوا عالی است.", "importance": 0.7, "similarity": 0.8, "topics": ["سلامت", "آب و هوا"]},
            {"content": "امیدوارم روز خوبی داشته باشید.", "importance": 0.6, "similarity": 0.75, "topics": ["سلامتی", "تفریح"]},
            {"content": "فکر می‌کنم شرایط بهتری خواهد شد.", "importance": 0.65, "similarity": 0.7, "topics": ["اقتصاد", "پیشرفت"]}
        ],
        "user_intent": {"intent_type": "INFORM", "topics": ["آب و هوا", "سلامت"]}
    }
    evaluator = Evaluator()
    quality = evaluator.evaluate_context_quality(sample_context)
    ranked_messages = evaluator.rank_context_messages(sample_context["context_messages"])
    report = evaluator.generate_context_report(sample_context)

    logger.info(f"Quality Score: {quality}")
    logger.info(f"Ranked Messages: {json.dumps(ranked_messages, ensure_ascii=False, indent=2)}")
    logger.info(f"Context Report: {json.dumps(report, ensure_ascii=False, indent=2)}")
