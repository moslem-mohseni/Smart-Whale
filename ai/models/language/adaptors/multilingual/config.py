CONFIG = {
    "use_teacher_model": True,  # اگر False شود، `teacher.py` دیگر اجرا نمی‌شود.
    "dependency_threshold": 0.01,  # اگر وابستگی زیر 1% باشد، معلم حذف می‌شود.
    "language": "multilingual",
    "tokenization": {
        "method": "basic",  # روش پیش‌فرض توکن‌سازی
    },
    "normalization": {
        "lowercase": True,  # نرمال‌سازی حروف کوچک
        "remove_punctuation": True,  # حذف نشانه‌گذاری
    },
    "sentiment_analysis": {
        "enabled": True,  # فعال یا غیرفعال کردن تحلیل احساسات
    },
    "semantics": {
        "vector_size": 768,  # اندازه بردار ویژگی برای پردازش معنایی
    }
}
