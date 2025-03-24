# persian/language_processors/proverbs/proverb_config.py
"""
ماژول proverb_config.py

این فایل شامل تنظیمات اختصاصی برای زیرسیستم ضرب‌المثل فارسی است.
تنظیمات شامل:
  - آستانه‌های شباهت برای تشخیص ضرب‌المثل‌ها.
  - تنظیمات بردارهای معنایی (ابعاد بردار).
  - تنظیمات Kafka (در صورت استفاده).
  - تنظیمات کش.
  - سایر پارامترهای مرتبط.
"""

CONFIG = {
    # تنظیمات شباهت
    "text_similarity_threshold": 0.7,
    "variant_similarity_threshold": 0.75,
    "detection_min_fragment_length": 3,

    # تنظیمات بردارهای معنایی
    "vector_dim": 128,

    # تنظیمات Kafka (در صورت استفاده)
    "kafka": {
        "producer_topic": "proverb_updates",
        "consumer_topic": "proverb_updates",
        "group_id": "proverb_processor"
    },

    # تنظیمات کش
    "cache_ttl": 86400,  # 24 ساعت

    # تنظیمات عمومی
    "confidence_threshold": 0.5
}

if __name__ == "__main__":
    import json

    print("Proverb configuration:")
    print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
