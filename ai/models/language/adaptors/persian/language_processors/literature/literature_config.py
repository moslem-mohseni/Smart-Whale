# persian/language_processors/literature/literature_config.py
"""
ماژول literature_config.py

این فایل شامل تنظیمات اختصاصی زیرسیستم ادبیات فارسی است.
تنظیمات شامل:
  - آستانه‌های شباهت متنی
  - تنظیمات مدل‌های هوشمند و معلم
  - TTL کش
  - ابعاد بردار معنایی
  - سایر پارامترهای مرتبط
"""

CONFIG = {
    "text_similarity_threshold": 0.7,
    "device_detection_threshold": 0.75,
    "meter_confidence_threshold": 0.6,
    "cache_ttl": 86400,  # 24 ساعت
    "vector_dim": 128,
    "model": {
        "use_smart_model": True,
        "use_teacher_model": True,
        "confidence_threshold": 0.5
    }
}

if __name__ == "__main__":
    import json
    print("Literature Configurations:")
    print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
