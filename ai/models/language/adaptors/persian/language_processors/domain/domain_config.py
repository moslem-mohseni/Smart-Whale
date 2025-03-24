# persian/language_processors/domain/domain_config.py

"""
ماژول domain_config.py

این فایل شامل تنظیمات اختصاصی زیرسیستم حوزه است.
تنظیمات شامل:
  - آستانه شباهت متنی (similarity_threshold)
  - تعداد حداقل تطبیق (min_match_count)
  - اندازه پنجره زمینه (context_window_size)
  - وزن مفاهیم (concept_weight) و روابط (relation_weight)
  - TTL کش (cache_ttl)
  - ابعاد بردار معنایی (vector_dim)
  - تنظیمات مدل (استفاده از مدل هوشمند و معلم، آستانه اطمینان)
"""

DOMAIN_CONFIG = {
    "similarity_threshold": 0.75,
    "min_match_count": 3,
    "context_window_size": 5,
    "concept_weight": 0.7,
    "relation_weight": 0.3,
    "cache_ttl": 3600,  # 1 ساعت
    "vector_dim": 128,
    "model": {
        "use_smart_model": True,
        "use_teacher_model": True,
        "confidence_threshold": 0.7
    }
}

if __name__ == "__main__":
    import json
    print("Domain Configurations:")
    print(json.dumps(DOMAIN_CONFIG, ensure_ascii=False, indent=2))
