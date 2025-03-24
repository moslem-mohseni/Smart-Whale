# persian/language_processors/contextual/conversation.py

"""
ماژول conversation.py

این فایل زیرسیستم مدیریت مکالمات در زبان فارسی را پیاده‌سازی می‌کند.
کلاس ConversationManager وظیفه ایجاد، به‌روزرسانی و بازیابی اطلاعات مکالمه را بر عهده دارد.
این زیرسیستم ورودی پردازشی (شناسه مکالمه، پیام فعلی و تاریخچه پیام‌ها) را دریافت و یک خروجی یکپارچه (Context)
شامل اطلاعات مکالمه، تاریخچه پیام‌ها، و اطلاعات زمینه‌ای (Intent, Topics, Quality) تولید می‌کند.
توجه: تمام وظایف مربوط به انتشار پیام‌ها، ذخیره‌سازی پایگاه داده و ثبت متریک‌ها در سطح بالاتر (مانند language_processor.py)
مدیریت می‌شود.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer
from ..utils.regex_utils import cleanup_text_for_compare

# تنظیم لاگ
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConversationManager:
    """
    کلاس ConversationManager مسئول مدیریت مکالمات فارسی در سطح زمینه (Context) است.
    این کلاس امکان ایجاد مکالمه، افزودن پیام، به‌روزرسانی وضعیت مکالمه و تولید خروجی یکپارچه زمینه را فراهم می‌کند.
    """

    def __init__(self):
        """
        مقداردهی اولیه ConversationManager.
        در اینجا ابزارهای پردازش متن مانند TextNormalizer و Tokenizer راه‌اندازی می‌شوند.
        همچنین حافظه داخلی (in-memory store) برای ذخیره‌ی اطلاعات مکالمه در سطح سیستم تعریف می‌شود.
        """
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer()
        # حافظه داخلی برای ذخیره مکالمات: {conversation_id: {metadata, messages, last_update}}
        self.conversations: Dict[str, Dict[str, Any]] = {}
        logger.info("ConversationManager راه‌اندازی شد.")

    def create_conversation(self, user_id: str, title: str = "", context_type: str = "CHAT",
                            description: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ایجاد یک مکالمه جدید با اطلاعات اولیه.

        Args:
            user_id (str): شناسه کاربر.
            title (str): عنوان مکالمه.
            context_type (str): نوع زمینه مکالمه (مانند CHAT، TASK، QA و ...).
            description (str): توضیحات مربوط به مکالمه.
            metadata (Optional[Dict[str, Any]]): اطلاعات تکمیلی (اختیاری).

        Returns:
            Dict[str, Any]: دیکشنری شامل اطلاعات مکالمه ایجادشده.
        """
        conversation_id = f"conv_{user_id}_{int(datetime.now().timestamp())}"
        if not title:
            title = f"مکالمه {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        conversation_data = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": title,
            "context_type": context_type,
            "description": description,
            "metadata": metadata if metadata else {},
            "start_time": datetime.now(),
            "last_update": datetime.now(),
            "messages": []  # تاریخچه پیام‌ها
        }
        self.conversations[conversation_id] = conversation_data
        logger.info(f"مکالمه جدید ایجاد شد: {conversation_id}")
        return conversation_data

    def add_message(self, conversation_id: str, user_id: str, role: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        افزودن پیام جدید به مکالمه موجود.

        Args:
            conversation_id (str): شناسه مکالمه.
            user_id (str): شناسه کاربر فرستنده.
            role (str): نقش فرستنده (مانند 'user' یا 'assistant').
            content (str): محتوای پیام.
            metadata (Optional[Dict[str, Any]]): اطلاعات تکمیلی پیام (اختیاری).

        Returns:
            Dict[str, Any]: اطلاعات پیام افزوده‌شده به مکالمه.
        """
        if conversation_id not in self.conversations:
            logger.error(f"مکالمه با شناسه {conversation_id} یافت نشد.")
            return {"success": False, "error": "Conversation not found"}

        # نرمال‌سازی محتوا
        normalized_content = self.normalizer.normalize(content)
        # ایجاد شناسه پیام
        message_id = f"msg_{conversation_id}_{int(datetime.now().timestamp())}"
        message_data = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "role": role,
            "content": normalized_content,
            "timestamp": datetime.now(),
            "metadata": metadata if metadata else {}
        }
        # افزودن پیام به تاریخچه مکالمه
        self.conversations[conversation_id]["messages"].append(message_data)
        self.conversations[conversation_id]["last_update"] = datetime.now()
        logger.info(f"پیام جدید به مکالمه {conversation_id} افزوده شد: {message_id}")
        return {"success": True, "message": message_data}

    def get_conversation_context(self, conversation_id: str, current_message: str, max_items: int = 10) -> Dict[
        str, Any]:
        """
        تولید یک خروجی یکپارچه زمینه مکالمه برای استفاده در تولید پاسخ.
        این تابع با استفاده از تاریخچه مکالمه و پیام فعلی، اطلاعات زمینه‌ای را استخراج و یک Context جامع برمی‌گرداند.

        Args:
            conversation_id (str): شناسه مکالمه.
            current_message (str): پیام فعلی که نیاز به پردازش دارد.
            max_items (int): حداکثر تعداد آیتم‌های زمینه‌ای.

        Returns:
            Dict[str, Any]: دیکشنری شامل اطلاعات زمینه (context) از جمله:
                - conversation_info: اطلاعات کلی مکالمه (عنوان، نوع، زمان شروع و تعداد پیام‌ها)
                - context_messages: پیام‌های انتخاب شده از تاریخچه به‌عنوان زمینه
                - user_intent: تحلیل نیت کاربر از پیام فعلی
                - related_entities: موجودیت‌های مرتبط استخراج‌شده (در صورت وجود)
                - evaluation: اطلاعات ارزیابی کیفیت زمینه مانند زمان پردازش و تعداد آیتم‌ها
        """
        if conversation_id not in self.conversations:
            logger.error(f"مکالمه با شناسه {conversation_id} یافت نشد.")
            return {"success": False, "error": "Conversation not found"}

        conversation = self.conversations[conversation_id]
        # نرمال‌سازی پیام فعلی برای تحلیل
        normalized_message = self.normalizer.normalize(current_message)

        # استخراج پیام‌های مرتبط از تاریخچه (به عنوان مثال، آخرین max_items پیام)
        messages = conversation.get("messages", [])
        context_messages = messages[-max_items:] if len(messages) >= max_items else messages

        # تحلیل نیت کاربر (در اینجا به صورت ساده با استفاده از توکنایزر و regex)
        intent = self._analyze_intent(normalized_message)

        # استخراج موجودیت‌های ساده (به صورت fallback)
        related_entities = self._extract_simple_entities(normalized_message)

        # ارزیابی کیفیت زمینه (به عنوان مثال، ترکیب تعداد پیام‌ها و میزان ارتباط)
        evaluation = {
            "processing_time": 0.0,  # این مقدار در زمان واقعی محاسبه می‌شود
            "context_items_count": len(context_messages) + len(related_entities)
        }

        context_output = {
            "conversation_info": {
                "conversation_id": conversation_id,
                "title": conversation.get("title", ""),
                "context_type": conversation.get("context_type", "CHAT"),
                "start_time": conversation.get("start_time").isoformat() if conversation.get("start_time") else "",
                "message_count": len(messages)
            },
            "context_messages": context_messages,
            "user_intent": intent,
            "related_entities": related_entities,
            "evaluation": evaluation,
            "success": True
        }

        logger.info(f"زمینه مکالمه {conversation_id} تولید شد.")
        return context_output

    def _analyze_intent(self, text: str) -> Dict[str, Any]:
        """
        تحلیل ساده نیت کاربر از متن پیام.
        این تابع می‌تواند با الگوریتم‌های پیچیده‌تری مانند مدل‌های NLP تعبیه‌شده بهبود یابد.
        در اینجا به صورت نمونه یک تحلیل ساده بر مبنای کلمات کلیدی انجام می‌شود.

        Args:
            text (str): متن نرمال‌شده پیام

        Returns:
            Dict[str, Any]: دیکشنری شامل نوع نیت (intent_type) و جزئیات اضافی
        """
        # به صورت ساده: اگر علامت سؤال وجود داشته باشد، نوع نیت QUESTION در نظر گرفته شود
        if "?" in text or "؟" in text:
            intent_type = "QUESTION"
        else:
            intent_type = "STATEMENT"
        # مثال ساده: استخراج موضوعات با استفاده از تقسیم کلمات
        tokens = text.split()
        topics = list(set(tokens[:3]))  # به عنوان نمونه، سه کلمه اول به عنوان موضوع
        return {
            "intent_type": intent_type,
            "topics": topics,
            "confidence": 0.7  # مقدار پیش‌فرض اطمینان
        }

    def _extract_simple_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        استخراج ساده موجودیت‌ها از متن با استفاده از الگوهای regex.
        در نسخه‌های پیشرفته می‌توان از مدل‌های استخراج موجودیت بهره برد.

        Args:
            text (str): متن نرمال‌شده

        Returns:
            List[Dict[str, Any]]: لیستی از موجودیت‌های استخراج‌شده
        """
        # به عنوان نمونه، استفاده از الگوی تشخیص ایمیل به عنوان موجودیت
        from ..utils.regex_utils import extract_pattern

        email_entities = extract_pattern(text, "EMAIL")
        # برگرداندن موجودیت‌ها به صورت یک لیست با ساختار ساده
        entities = []
        for item in email_entities:
            entities.append({
                "entity_type": "EMAIL",
                "entity_text": item.get("match"),
                "start": item.get("start"),
                "end": item.get("end"),
                "confidence": 0.9
            })
        return entities

    # متدهای تکمیلی برای تعاملات مکالمه می‌توانند به مرور اضافه شوند...


if __name__ == "__main__":
    # تست نمونه برای ایجاد مکالمه، افزودن پیام و تولید زمینه
    cm = ConversationManager()
    conv = cm.create_conversation(user_id="user_123", title="گفتگوی نمونه", context_type="CHAT")
    cm.add_message(conv["conversation_id"], "user_123", "user", "سلام! امروز هوا عالی است.")
    cm.add_message(conv["conversation_id"], "user_123", "user", "آیا می‌توانید توضیح دهید که چرا اینقدر خوب است؟")

    context = cm.get_conversation_context(conv["conversation_id"], "چرا هوا اینقدر عالیه؟", max_items=5)
    print("زمینه مکالمه:", context)
