CONFIG = {
    "use_hazm": True,  # استفاده از hazm تا زمانی که مدل Smart Whale به اندازه کافی قوی شود
    "confidence_threshold": 0.85,  # آستانه اعتماد برای پردازش مستقل توسط مدل
    "cache_enabled": True,  # فعال‌سازی کشینگ در Redis برای افزایش سرعت پردازش
    "vector_store_enabled": True,  # استفاده از Vector Search برای پردازش معنایی
    "database_storage": "clickhouse",  # نوع پایگاه داده اصلی برای ذخیره‌سازی اطلاعات
    "language_model": "parsBERT",  # مدل پیش‌فرض مورد استفاده برای پردازش زبان فارسی
    "teacher_model": "teacher.pth",  # فایل ذخیره‌سازی مدل معلم
    "smart_model": "smart_model.pth",  # فایل ذخیره‌سازی مدل یادگیرنده Smart Whale
    "enable_logging": True  # فعال‌سازی ثبت لاگ برای تحلیل عملکرد سیستم
}
