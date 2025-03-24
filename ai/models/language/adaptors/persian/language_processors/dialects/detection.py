# language_processors/dialects/detection.py

import json
import re
import time
import logging
from collections import defaultdict
from typing import Dict, Any, Optional

# استفاده از کلاس نرمالایزر موجود در utils
from ..utils.text_normalization import TextNormalizer

logger = logging.getLogger("Detection")
logger.setLevel(logging.INFO)


def get_standard_dialect(data_access) -> Dict[str, Any]:
    """
    دریافت لهجه استاندارد (فارسی معیار) از داده‌های ذخیره شده.
    """
    dialects = data_access.storage.get("dialects", {})
    for dialect in dialects.values():
        if dialect.get("dialect_code", "").upper() == "STANDARD":
            return dialect
    # در صورت عدم وجود، مقدار پیش‌فرض ارائه می‌شود.
    return {"dialect_id": "d_1001", "dialect_name": "فارسی معیار", "dialect_code": "STANDARD"}


def rule_based_dialect_detection(text: str, data_access, detection_params: dict) -> Dict[str, Any]:
    """
    تشخیص لهجه بر اساس الگوریتم قاعده‌محور با استفاده از ویژگی‌ها و واژگان لهجه‌ای.

    Args:
        text (str): متن نرمال‌شده
        data_access: شیء دسترسی به داده‌ها (شامل dialects، features و words)
        detection_params (dict): پارامترهای تشخیص مانند similarity_threshold، feature_weight و word_weight

    Returns:
        dict: نتیجه تشخیص لهجه شامل شناسه، نام، کد، سطح اطمینان و جزئیات مربوط به ویژگی‌ها و واژگان یافت شده
    """
    dialect_features = data_access.storage.get("features", {})
    dialect_words = data_access.storage.get("words", {})
    dialects = data_access.storage.get("dialects", {})

    scores = defaultdict(lambda: {"score": 0, "features": [], "words": []})

    # امتیازدهی بر مبنای ویژگی‌ها
    for feature_id, feature in dialect_features.items():
        dialect_id = feature.get("dialect_id")
        pattern = feature.get("feature_pattern")
        if not dialect_id or not pattern:
            continue
        try:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                weight = feature.get("confidence", 0.8) * detection_params.get("feature_weight", 0.7)
                increment = weight * len(matches)
                scores[dialect_id]["score"] += increment
                scores[dialect_id]["features"].append({
                    "feature_id": feature_id,
                    "type": feature.get("feature_type"),
                    "description": feature.get("description"),
                    "matches": [m.group(0) for m in matches],
                    "count": len(matches)
                })
        except Exception as e:
            logger.error(f"خطا در پردازش ویژگی {feature_id}: {e}")

    # امتیازدهی بر مبنای واژگان
    for word_id, word_info in dialect_words.items():
        dialect_id = word_info.get("dialect_id")
        word = word_info.get("word")
        if not dialect_id or not word:
            continue
        try:
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                weight = word_info.get("confidence", 0.8) * detection_params.get("word_weight", 0.3)
                increment = weight * len(matches)
                scores[dialect_id]["score"] += increment
                scores[dialect_id]["words"].append({
                    "word_id": word_id,
                    "word": word,
                    "standard": word_info.get("standard_equivalent"),
                    "count": len(matches)
                })
        except Exception as e:
            logger.error(f"خطا در پردازش واژه {word_id}: {e}")

    # انتخاب لهجه با بالاترین امتیاز
    max_score = 0
    selected_dialect = None
    for did, data in scores.items():
        if data["score"] > max_score:
            max_score = data["score"]
            selected_dialect = did

    threshold = detection_params.get("similarity_threshold", 0.75)
    if not selected_dialect or max_score < threshold:
        std = get_standard_dialect(data_access)
        return {
            "dialect_id": std.get("dialect_id", "d_1001"),
            "dialect_name": std.get("dialect_name", "فارسی معیار"),
            "dialect_code": std.get("dialect_code", "STANDARD"),
            "confidence": 0.6,
            "features_found": [],
            "words_found": [],
            "source": "rule_based"
        }

    confidence = min(max_score, 1)
    if selected_dialect in dialects:
        result = {
            "dialect_id": selected_dialect,
            "dialect_name": dialects[selected_dialect].get("dialect_name", "نامشخص"),
            "dialect_code": dialects[selected_dialect].get("dialect_code", "UNKNOWN"),
            "confidence": confidence,
            "features_found": scores[selected_dialect]["features"],
            "words_found": scores[selected_dialect]["words"],
            "source": "rule_based"
        }
    else:
        result = {
            "dialect_id": selected_dialect,
            "dialect_name": "نامشخص",
            "dialect_code": "UNKNOWN",
            "confidence": confidence,
            "features_found": scores[selected_dialect]["features"],
            "words_found": scores[selected_dialect]["words"],
            "source": "rule_based"
        }
    return result


def detect_dialect(text: str, data_access, config: dict) -> Dict[str, Any]:
    """
    تشخیص پیشرفته لهجه فارسی با استفاده از چندین استراتژی:
      1. استفاده از مدل معلم (Teacher) به عنوان راهنما.
      2. استفاده از مدل دانش‌آموز (Smart Model) در صورت اطمینان کافی.
      3. در نهایت استفاده از روش قاعده‌محور.

    علاوه بر این، نتیجه تشخیص در کش ذخیره و متریک‌های عملکرد ثبت می‌شود.

    Args:
        text (str): متن ورودی (ترجیحاً نرمال‌شده)
        data_access: شیء دسترسی به داده‌ها (حاوی dialects، features، words و غیره)
        config (dict): تنظیمات شامل موارد زیر:
            - "cache_manager": شیء مدیریت کش
            - "performance_metrics": شیء جمع‌آوری متریک‌ها
            - "teacher": مدل معلم (در صورت موجود بودن)
            - "smart_model": مدل دانش‌آموز
            - "confidence_threshold": آستانه اطمینان
            - "detection_params": پارامترهای تشخیص (مانند similarity_threshold، feature_weight و word_weight)

    Returns:
        dict: دیکشنری نتیجه شامل dialect_id، dialect_name، dialect_code، سطح اطمینان و منبع تشخیص
    """
    # استفاده از TextNormalizer از utils جهت نرمال‌سازی متن
    normalizer = TextNormalizer()
    normalized_text = normalizer.normalize(text)

    cache_manager = config.get("cache_manager")
    cache_key = f"dialect_detection:{normalized_text}"
    if cache_manager:
        cached = cache_manager.get_cached_result(cache_key)
        if cached:
            if "performance_metrics" in config:
                config["performance_metrics"].collect_metrics({
                    "dialect_detection": {
                        "text_length": len(normalized_text),
                        "detected_dialect": "Cached",
                        "confidence": 1.0,
                        "source": "cache"
                    }
                })
            return json.loads(cached)

    result = {}
    teacher = config.get("teacher")
    smart_model = config.get("smart_model")
    confidence_threshold = config.get("confidence_threshold", 0.8)

    # استفاده از مدل معلم (Teacher) به عنوان راهنما
    if teacher:
        try:
            teacher_result = teacher.detect_dialect(normalized_text)
            if teacher_result and teacher_result.get("dialect_id"):
                teacher_result["source"] = "teacher"
                result = teacher_result
                # استفاده از خروجی معلم جهت آموزش مدل دانش‌آموز (Smart Model)
                if smart_model:
                    try:
                        smart_model.learn_from_teacher(normalized_text, teacher_result)
                    except Exception as learn_e:
                        logger.error(f"خطا در یادگیری از معلم: {learn_e}")
        except Exception as e:
            logger.error(f"خطا در تشخیص لهجه با مدل معلم: {e}")

    # اگر نتیجه معلم به دست نیامد یا سطح اطمینان پایین بود، استفاده از دانش‌آموز (Smart Model)
    if (not result or "dialect_id" not in result) and smart_model:
        try:
            smart_conf = smart_model.confidence_level(normalized_text)
        except Exception as e:
            logger.error(f"خطا در دریافت سطح اطمینان از دانش‌آموز: {e}")
            smart_conf = 0.0
        if smart_conf >= confidence_threshold:
            try:
                smart_result = smart_model.detect_dialect(normalized_text)
                if smart_result and smart_result.get("dialect_id"):
                    smart_result["source"] = "smart_model"
                    result = smart_result
            except Exception as e:
                logger.error(f"خطا در تشخیص لهجه با دانش‌آموز: {e}")

    # اگر هیچ مدل نتوانست نتیجه ارائه دهد، از روش قاعده‌محور استفاده می‌کنیم.
    if not result or "dialect_id" not in result:
        detection_params = config.get("detection_params", {
            "similarity_threshold": 0.75,
            "feature_weight": 0.7,
            "word_weight": 0.3
        })
        result = rule_based_dialect_detection(normalized_text, data_access, detection_params)

    # افزودن اطلاعات تکمیلی لهجه از داده‌های ذخیره شده
    dialects = data_access.storage.get("dialects", {})
    if result.get("dialect_id") in dialects:
        dialect_info = dialects[result["dialect_id"]]
        result["dialect_name"] = dialect_info.get("dialect_name", "")
        result["dialect_code"] = dialect_info.get("dialect_code", "")
        result["region"] = dialect_info.get("region", "")

    # ذخیره نتیجه در کش
    if cache_manager:
        cache_manager.cache_result(cache_key, json.dumps(result), 86400)

    # ثبت متریک عملکرد
    performance_metrics = config.get("performance_metrics")
    if performance_metrics:
        performance_metrics.collect_metrics({
            "dialect_detection": {
                "text_length": len(normalized_text),
                "detected_dialect": result.get("dialect_name", "نامشخص"),
                "confidence": result.get("confidence", 0.0),
                "source": result.get("source", "unknown")
            }
        })

    logger.info(f"تشخیص لهجه به پایان رسید: {result.get('dialect_name', 'نامشخص')} (منبع: {result.get('source')})")
    return result


# نمونه تست برای این ماژول
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)


    # تعریف کلاس‌های dummy برای cache_manager و performance_metrics جهت تست
    class DummyCacheManager:
        def __init__(self):
            self.cache = {}

        def get_cached_result(self, key: str) -> Optional[str]:
            return self.cache.get(key)

        def cache_result(self, key: str, value, ttl: int):
            self.cache[key] = value


    class DummyPerformanceMetrics:
        def collect_metrics(self, metrics: dict):
            logger.info(f"Metrics: {metrics}")


    # تعریف داده‌های نمونه (in-memory) برای DataAccess
    class DummyDataAccess:
        def __init__(self):
            self.storage = {
                "dialects": {
                    "d_1001": {"dialect_id": "d_1001", "dialect_name": "فارسی معیار", "dialect_code": "STANDARD",
                               "region": "سراسر ایران"}
                },
                "features": {
                    "f_1": {"feature_id": "f_1", "dialect_id": "d_1001", "feature_type": "PHONETIC",
                            "feature_pattern": r"\b(سلام)\b", "description": "تشخیص سلام", "confidence": 0.9}
                },
                "words": {
                    "w_1": {"word_id": "w_1", "dialect_id": "d_1001", "word": "سلام", "standard_equivalent": "سلام",
                            "confidence": 0.9}
                }
            }


    # برای تست، فرض کنید teacher و smart_model در دسترس نباشند.
    dummy_config = {
        "cache_manager": DummyCacheManager(),
        "performance_metrics": DummyPerformanceMetrics(),
        "detection_params": {
            "similarity_threshold": 0.75,
            "feature_weight": 0.7,
            "word_weight": 0.3
        },
        "confidence_threshold": 0.8,
        "teacher": None,
        "smart_model": None
    }

    dummy_data_access = DummyDataAccess()

    sample_text = "سلام، این یک متن آزمایشی است."
    detected_result = detect_dialect(sample_text, dummy_data_access, dummy_config)
    print("نتیجه تشخیص لهجه:", json.dumps(detected_result, ensure_ascii=False, indent=2))
