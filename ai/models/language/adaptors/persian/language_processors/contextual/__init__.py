"""
پوشه contextual شامل زیرسیستم‌های مرتبط با مدیریت زمینه (Context) زبان فارسی است.
این زیرسیستم‌ها شامل تنظیمات، ابزارهای NLP، مدیریت مکالمه، پردازش پیام، تحلیل نیت کاربر،
ارزیابی کیفیت زمینه و توابع کمکی عمومی می‌باشند.

ماژول‌های صادرشده در این پوشه:
    - config: شامل توابعی جهت دسترسی به ثابت‌ها و تنظیمات زمینه (Context Types, Knowledge Types, …)
    - conversation: کلاس ConversationManager برای مدیریت مکالمات (ایجاد، به‌روزرسانی و بازیابی)
    - evaluator: کلاس Evaluator جهت ارزیابی کیفیت زمینه تولید شده
    - intent_analyzer: کلاس IntentAnalyzer جهت تحلیل نیت پیام‌ها
    - message_processor: کلاس MessageProcessor جهت پردازش پیام‌ها و استخراج اطلاعات زمینه‌ای
    - nlp_tools: کلاس NLPEngine برای فراهم کردن ابزارهای پردازش زبان طبیعی (نرمال‌سازی و توکنیزیشن)
    - utils: توابع و ابزارهای کمکی عمومی جهت پردازش متن (مانند توابع regex و …)

توسط این فایل می‌توان به‌سادگی به تمام امکانات زیرسیستم contextual دسترسی پیدا کرد.
"""

from .config import (
    get_context_types,
    get_knowledge_types,
    get_relation_types,
    get_importance_levels,
    get_confidence_levels
)
from .conversation import ConversationManager
from .evaluator import Evaluator
from .intent_analyzer import IntentAnalyzer
from .message_processor import MessageProcessor
from .nlp_tools import NLPEngine

__all__ = [
    "get_context_types",
    "get_knowledge_types",
    "get_relation_types",
    "get_importance_levels",
    "get_confidence_levels",
    "ConversationManager",
    "Evaluator",
    "IntentAnalyzer",
    "MessageProcessor",
    "NLPEngine"
]
