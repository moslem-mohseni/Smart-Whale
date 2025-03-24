"""
ماژول message_processor.py

این ماژول به عنوان زیرسیستم پردازش زمینه زبان فارسی عمل می‌کند.
ورودی این ماژول شامل شناسه مکالمه، پیام فعلی و تاریخچه پیام‌ها (به صورت لیست دیکشنری) می‌باشد.
خروجی نهایی یک دیکشنری یکپارچه شامل اطلاعات زیر است:
    - conversation_info: اطلاعات کلی مکالمه (شناسه، تعداد پیام‌ها و سایر اطلاعات پایه)
    - context_messages: لیستی از پیام‌های مرتبط به عنوان زمینه
    - user_intent: تحلیل نیت کاربر از پیام فعلی
    - related_knowledge: دانش‌های استخراج‌شده مرتبط با پیام
    - related_entities: موجودیت‌های استخراج‌شده مرتبط با پیام
    - evaluation: اطلاعات ارزیابی مانند زمان پردازش، کیفیت زمینه و تعداد آیتم‌های استخراج‌شده
    - success: وضعیت موفقیت عملیات

توجه: در این نسخه تمامی عملیات انتشار، ذخیره‌سازی پایگاه داده و ثبت متریک به سطح بالاتر (language_processor) واگذار شده است.
"""

import json
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from collections import Counter
from typing import List, Dict, Any

# واردات تنظیمات زمینه از ماژول config در همین پوشه
from .config import (
    get_context_types,
    get_knowledge_types,
    get_relation_types,
    get_importance_levels,
    get_confidence_levels
)

# واردات ابزارهای NLP از پوشه utils
from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer
from ..utils.regex_utils import extract_all_patterns  # در صورت نیاز به الگوهای regex

# تنظیمات مرکزی زمینه (CONFIG) با استفاده از توابع config
CONFIG = {
    "context_types": get_context_types(),
    "knowledge_types": get_knowledge_types(),
    "relation_types": get_relation_types(),
    "importance_levels": get_importance_levels(),
    "confidence_levels": get_confidence_levels(),
    "max_history_length": 50,
    "context_window_size": 8,
    "relevance_threshold": 0.75
}


class MessageProcessor:
    """
    کلاس MessageProcessor

    این کلاس تمامی عملیات پردازش پیام فعلی و استخراج اطلاعات زمینه‌ای را انجام می‌دهد.
    عملیات شامل نرمال‌سازی متن، تحلیل نیت کاربر، رتبه‌بندی پیام‌های تاریخچه، استخراج دانش و موجودیت‌های مرتبط،
    و ارزیابی کیفیت نهایی زمینه است.
    """

    def __init__(self):
        """راه‌اندازی ابزارهای NLP مورد نیاز با استفاده از کلاس‌های TextNormalizer و Tokenizer."""
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer()

    def process(self, conversation_id: str, current_message: str, previous_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        پردازش پیام فعلی به همراه تاریخچه پیام‌ها و استخراج اطلاعات زمینه‌ای.

        Args:
            conversation_id (str): شناسه مکالمه.
            current_message (str): متن پیام فعلی.
            previous_messages (List[Dict[str, Any]]): لیستی از پیام‌های قبلی (هر کدام دیکشنری شامل 'message_id'، 'content' و 'timestamp').

        Returns:
            Dict[str, Any]: دیکشنری شامل:
                - conversation_info: اطلاعات کلی مکالمه.
                - context_messages: لیست پیام‌های مرتبط.
                - user_intent: تحلیل نیت کاربر.
                - related_knowledge: دانش‌های استخراج‌شده.
                - related_entities: موجودیت‌های استخراج‌شده.
                - evaluation: ارزیابی کیفیت زمینه (زمان پردازش، کیفیت، تعداد آیتم‌ها).
                - success: True در صورت موفقیت.
        """
        start_time = time.time()

        # نرمال‌سازی پیام فعلی
        normalized_message = self.normalizer.normalize(current_message)

        # تحلیل نیت کاربر
        user_intent = self._analyze_user_intent(normalized_message, previous_messages)

        # رتبه‌بندی پیام‌های قبلی بر اساس شباهت محتوایی با پیام فعلی
        ranked_messages = self._rank_messages_by_relevance(previous_messages, normalized_message)
        context_messages = self._select_context_messages(ranked_messages, max_items=CONFIG["context_window_size"])

        # استخراج دانش مرتبط با پیام (مثلاً تاریخ، شماره تلفن، ایمیل و غیره)
        related_knowledge = self._get_related_knowledge(normalized_message)

        # استخراج موجودیت‌های مرتبط (با استفاده از الگوهای regex موجود در regex_utils)
        related_entities = self._get_related_entities(normalized_message)

        # ترکیب اطلاعات مکالمه
        conversation_info = {
            "conversation_id": conversation_id,
            "message_count": len(previous_messages)
        }

        processing_time = time.time() - start_time
        context_quality = self._evaluate_context_quality(context_messages, related_knowledge, related_entities)

        result = {
            "conversation_info": conversation_info,
            "context_messages": context_messages,
            "user_intent": user_intent,
            "related_knowledge": related_knowledge,
            "related_entities": related_entities,
            "evaluation": {
                "processing_time": round(processing_time, 4),
                "context_quality": context_quality,
                "context_items_count": len(context_messages) + len(related_knowledge) + len(related_entities)
            },
            "success": True
        }
        return result

    def _analyze_user_intent(self, current_message: str, previous_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        تحلیل نیت کاربر از پیام فعلی با استفاده از الگوریتم‌های ساده مبتنی بر الگوهای متنی.

        Args:
            current_message (str): متن پیام فعلی (نرمال‌شده).
            previous_messages (List[Dict[str, Any]]): لیست پیام‌های قبلی.

        Returns:
            Dict[str, Any]: اطلاعات نیت شامل 'intent_type'، 'topics' و 'urgency'.
        """
        intent = {}
        # تشخیص نیت بر اساس علامت‌های پرسشی
        if "?" in current_message or "؟" in current_message:
            intent["intent_type"] = "QUESTION"
        elif any(word in current_message.lower() for word in ["لطفاً", "خواهش"]):
            intent["intent_type"] = "REQUEST"
        else:
            intent["intent_type"] = "STATEMENT"

        # استخراج موضوعات (به صورت ساده با استفاده از توکنایزر)
        words = self.tokenizer.tokenize_words(current_message)
        common_words = Counter(words).most_common(3)
        topics = [word for word, _ in common_words if len(word) > 3]
        intent["topics"] = topics

        # تعیین فوریت پیام به صورت ساده
        intent["urgency"] = 0.5 if any(urgent in current_message.lower() for urgent in ["فوری", "عجله"]) else 0.3

        return intent

    def _rank_messages_by_relevance(self, messages: List[Dict[str, Any]], current_message: str) -> List[Dict[str, Any]]:
        """
        رتبه‌بندی پیام‌های قبلی بر اساس شباهت محتوایی با پیام فعلی.

        Args:
            messages (List[Dict[str, Any]]): لیست پیام‌های قبلی.
            current_message (str): متن پیام فعلی.

        Returns:
            List[Dict[str, Any]]: لیستی از پیام‌ها مرتب‌شده به ترتیب نزولی شباهت.
        """
        ranked = []
        for msg in messages:
            content = msg.get("content", "")
            similarity = SequenceMatcher(None, current_message, content).ratio()
            ranked.append((msg, similarity))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [msg for msg, _ in ranked]

    def _select_context_messages(self, ranked_messages: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
        """
        انتخاب حداکثر تعداد مشخصی از پیام‌های مرتبط به عنوان زمینه.

        Args:
            ranked_messages (List[Dict[str, Any]]): پیام‌های رتبه‌بندی‌شده.
            max_items (int): حداکثر تعداد پیام‌های انتخابی.

        Returns:
            List[Dict[str, Any]]: لیست پیام‌های انتخاب‌شده.
        """
        return ranked_messages[:max_items]

    def _get_related_knowledge(self, current_message: str) -> List[Dict[str, Any]]:
        """
        استخراج دانش مرتبط با پیام فعلی با استفاده از توابع موجود در regex_utils.

        Args:
            current_message (str): متن پیام فعلی.

        Returns:
            List[Dict[str, Any]]: لیستی از دانش‌های استخراج‌شده (مثلاً تاریخ‌ها، مبالغ و درصدها).
        """
        knowledge = []
        # استفاده از الگوهای موجود در regex_utils برای استخراج تاریخ شمسی
        patterns = ["DATE_PERSIAN", "MONEY", "PERCENT"]
        all_patterns = extract_all_patterns(current_message, pattern_keys=patterns)
        for key, matches in all_patterns.items():
            for match in matches:
                knowledge.append({
                    "type": key,
                    "value": match["match"],
                    "confidence": 0.9  # مقدار اطمینان پیش‌فرض
                })
        return knowledge

    def _get_related_entities(self, current_message: str) -> List[Dict[str, Any]]:
        """
        استخراج موجودیت‌های مرتبط از پیام فعلی با استفاده از الگوهای موجود در regex_utils.

        Args:
            current_message (str): متن پیام فعلی.

        Returns:
            List[Dict[str, Any]]: لیست موجودیت‌های استخراج‌شده (مانند ایمیل‌ها، URLها و شماره تلفن).
        """
        entities = []
        # استفاده از الگوی EMAIL به عنوان نمونه
        email_matches = extract_all_patterns(current_message, pattern_keys=["EMAIL"]).get("EMAIL", [])
        for match in email_matches:
            entities.append({
                "type": "EMAIL",
                "value": match["match"],
                "confidence": 0.95
            })
        return entities

    def _evaluate_context_quality(self, context_messages: List[Dict[str, Any]], knowledge: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> float:
        """
        ارزیابی کیفیت زمینه استخراج‌شده بر اساس تعداد آیتم‌های استخراج‌شده.

        Args:
            context_messages (List[Dict[str, Any]]): پیام‌های انتخاب‌شده به عنوان زمینه.
            knowledge (List[Dict[str, Any]]): دانش‌های استخراج‌شده.
            entities (List[Dict[str, Any]]): موجودیت‌های استخراج‌شده.

        Returns:
            float: امتیاز کیفیت (بین 0 و 1).
        """
        total_items = len(context_messages) + len(knowledge) + len(entities)
        if total_items == 0:
            return 0.0
        quality = min(total_items / 10.0, 1.0)
        return round(quality, 2)


# نمونه استفاده جهت تست مستقل ماژول
if __name__ == "__main__":
    processor = MessageProcessor()
    conversation_id = "conv_test_001"
    current_message = "سلام، لطفاً توضیح بده امروز چه اتفاقی افتاد؟"
    previous_messages = [
        {"message_id": "msg_001", "content": "سلام، خوبی؟", "timestamp": datetime.now()},
        {"message_id": "msg_002", "content": "امروز هوا خیلی سرد بود.", "timestamp": datetime.now()},
        {"message_id": "msg_003", "content": "به نظرم باران میاد.", "timestamp": datetime.now()},
    ]

    context = processor.process(conversation_id, current_message, previous_messages)
    print(json.dumps(context, ensure_ascii=False, indent=4))
