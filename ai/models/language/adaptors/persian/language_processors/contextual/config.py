# persian/language_processors/utils/config.py
"""
این ماژول ثابت‌ها و تنظیمات اولیه مربوط به مدیریت زمینه (Context) زبان فارسی را تعریف می‌کند.
ثابت‌ها شامل انواع زمینه‌های مکالمه، انواع دانش استخراج‌شده، انواع روابط بین عناصر مکالمه،
سطوح اهمیت و سطوح اطمینان به استنتاج هستند.
"""

from typing import Dict

# انواع زمینه‌های مکالمه: توضیح کوتاه در کنار هر کدام
CONTEXT_TYPES: Dict[str, str] = {
    "CHAT": "گفتگوی معمولی",
    "TASK": "انجام وظیفه",
    "QA": "پرسش و پاسخ",
    "TUTORING": "آموزش",
    "BRAINSTORMING": "طوفان فکری",
    "STORY": "داستان‌گویی",
    "DEBATE": "بحث و مناظره"
}

# انواع دانش استخراج‌شده: دسته‌بندی‌های مختلف دانش
KNOWLEDGE_TYPES: Dict[str, str] = {
    "FACTUAL": "دانش واقعی",
    "PREFERENCE": "ترجیحات کاربر",
    "PERSONAL": "اطلاعات شخصی",
    "DOMAIN": "دانش تخصصی",
    "WORLD": "دانش عمومی جهانی",
    "LINGUISTIC": "دانش زبانی",
    "CULTURAL": "دانش فرهنگی"
}

# انواع روابط بین عناصر مکالمه: نشان‌دهنده ارتباط معنایی یا ساختاری بین پیام‌ها
RELATION_TYPES: Dict[str, str] = {
    "ELABORATION": "بسط",
    "CONTRAST": "تضاد",
    "CAUSE": "علت",
    "EFFECT": "معلول",
    "TEMPORAL": "زمانی",
    "CORRECTION": "تصحیح",
    "REPETITION": "تکرار",
    "QUESTION": "پرسش",
    "ANSWER": "پاسخ",
    "EXAMPLE": "مثال"
}

# سطوح مختلف اهمیت زمینه: برای مشخص کردن میزان اهمیت اطلاعات زمینه‌ای
IMPORTANCE_LEVELS: Dict[str, str] = {
    "CRITICAL": "ضروری",
    "HIGH": "بالا",
    "MEDIUM": "متوسط",
    "LOW": "پایین",
    "BACKGROUND": "پس‌زمینه"
}

# سطوح اطمینان به استنتاج: برای ارزیابی اعتماد به نتایج استخراج شده از متن
CONFIDENCE_LEVELS: Dict[str, str] = {
    "VERY_HIGH": "بسیار بالا",
    "HIGH": "بالا",
    "MEDIUM": "متوسط",
    "LOW": "پایین",
    "VERY_LOW": "بسیار پایین"
}

# توابع کمکی جهت دسترسی آسان به تنظیمات

def get_context_types() -> Dict[str, str]:
    """بازگشت انواع زمینه‌های مکالمه."""
    return CONTEXT_TYPES

def get_knowledge_types() -> Dict[str, str]:
    """بازگشت انواع دانش استخراج‌شده."""
    return KNOWLEDGE_TYPES

def get_relation_types() -> Dict[str, str]:
    """بازگشت انواع روابط بین عناصر مکالمه."""
    return RELATION_TYPES

def get_importance_levels() -> Dict[str, str]:
    """بازگشت سطوح اهمیت زمینه."""
    return IMPORTANCE_LEVELS

def get_confidence_levels() -> Dict[str, str]:
    """بازگشت سطوح اطمینان به استنتاج."""
    return CONFIDENCE_LEVELS


if __name__ == "__main__":
    # تست ساده برای چاپ مقادیر تنظیمات
    print("Context Types:", get_context_types())
    print("Knowledge Types:", get_knowledge_types())
    print("Relation Types:", get_relation_types())
    print("Importance Levels:", get_importance_levels())
    print("Confidence Levels:", get_confidence_levels())
