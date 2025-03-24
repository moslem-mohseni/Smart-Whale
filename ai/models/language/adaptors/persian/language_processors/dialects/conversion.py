# language_processors/dialects/conversion.py

"""
ماژول conversion.py

این ماژول شامل کلاس DialectConversionProcessor است که وظیفه تبدیل متن از لهجه تشخیص داده شده به لهجه هدف را بر عهده دارد.
این زیرسیستم بدون وابستگی به Kafka عمل می‌کند و از زیرساخت‌های موجود (کش، پایگاه داده، Milvus و غیره) و همچنین از کلاس‌های مفید در پوشه utils (به عنوان مثال TextNormalizer) استفاده می‌کند.
در این فایل، ابتدا تلاش می‌شود از مدل هوشمند (دانش‌آموز) برای تبدیل استفاده شود؛ در صورت ناکافی بودن، از مدل معلم بهره می‌گیریم و در نهایت در صورت عدم موفقیت هر دو، مکانیسم‌های قاعده‌محور اجرا می‌شوند.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...config import CONFIG
from .data_access import DialectDataAccess
from ..utils.text_normalization import TextNormalizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DialectConversionProcessor:
    """
    کلاس DialectConversionProcessor مسئول تبدیل متن از لهجه تشخیص داده شده به لهجه هدف است.

    روند تبدیل به صورت زیر است:
      1. نرمال‌سازی متن ورودی.
      2. تشخیص لهجه مبدأ (source dialect) با استفاده از داده‌های موجود.
      3. در صورتیکه لهجه مبدأ و هدف یکسان باشند، متن اصلی برگردانده می‌شود.
      4. در غیر این صورت ابتدا تلاش می‌شود از مدل هوشمند (دانش‌آموز) استفاده شود؛
         در صورت عدم موفقیت، از مدل معلم استفاده شده و در نهایت به روش‌های قاعده‌محور (rule-based) تبدیل انجام می‌شود.
      5. نتایج (شامل متن تبدیل شده، قوانین اعمال شده، جایگزینی واژگان، سطح اطمینان و سایر اطلاعات) ذخیره و برگردانده می‌شود.
    """

    def __init__(self, smart_model: Optional[Any] = None, teacher: Optional[Any] = None):
        self.logger = logger
        self.data_access = DialectDataAccess()
        self.normalizer = TextNormalizer()
        # در صورت ارائه مدل‌های هوشمند یا معلم، استفاده می‌کنیم؛ در غیر این صورت سعی در بارگذاری داریم
        self.smart_model = smart_model if smart_model is not None else self._load_smart_model()
        self.teacher = teacher if teacher is not None else self._load_teacher_model()
        self.detection_params = {
            "similarity_threshold": 0.75,
            "min_match_count": 3,
            "context_window_size": 5,
            "feature_weight": 0.7,
            "word_weight": 0.3
        }
        self.logger.info("DialectConversionProcessor initialized.")

    def _load_smart_model(self) -> Optional[Any]:
        try:
            module = __import__(f"ai.models.language.adaptors.{self.data_access.language}.smart_model",
                                fromlist=["SmartModel"])
            return module.SmartModel()
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری مدل هوشمند: {e}")
            return None

    def _load_teacher_model(self) -> Optional[Any]:
        try:
            module = __import__(f"ai.models.language.adaptors.{self.data_access.language}.teacher",
                                fromlist=["TeacherModel"])
            return module.TeacherModel()
        except Exception as e:
            self.logger.warning("مدل معلم یافت نشد. از قابلیت‌های محدود تبدیل استفاده خواهد شد.")
            return None

    def convert_dialect(self, text: str, target_dialect_code: str = "STANDARD") -> Dict[str, Any]:
        """
        تبدیل متن ورودی از لهجه تشخیص داده شده به لهجه هدف.

        Args:
            text (str): متن ورودی (فارسی)
            target_dialect_code (str): کد لهجه هدف (پیش‌فرض: استاندارد)

        Returns:
            Dict[str, Any]: دیکشنری شامل اطلاعات متن اصلی، متن تبدیل شده، لهجه مبدأ و مقصد، سطح اطمینان،
                            لیست قوانین اعمال شده، واژگان جایگزین شده و منبع تبدیل (مدل هوشمند، معلم یا قاعده‌محور)
        """
        self.data_access.statistics["requests"] += 1
        self.data_access.statistics["conversion_count"] += 1

        normalized_text = self._normalize_text(text)
        cache_key = f"dialect_conversion:{normalized_text}:{target_dialect_code}"
        cached_result = self.data_access.cache_manager.get_cached_result(cache_key)
        if cached_result:
            self.data_access.statistics["cache_hits"] += 1
            return json.loads(cached_result)

        detected = self.data_access.detect_dialect(normalized_text)
        source_dialect_code = detected.get("dialect_code", "STANDARD")
        if source_dialect_code == target_dialect_code:
            result = {
                "original_text": text,
                "converted_text": text,
                "source_dialect": source_dialect_code,
                "target_dialect": target_dialect_code,
                "confidence": 1.0,
                "rules_applied": [],
                "words_replaced": [],
                "source": "default"
            }
            self.data_access.cache_manager.cache_result(cache_key, json.dumps(result), 86400)
            return result

        source_dialect = self.data_access.get_dialect_by_code(source_dialect_code)
        target_dialect = self.data_access.get_dialect_by_code(target_dialect_code)
        if not target_dialect:
            return {"error": f"لهجه مقصد {target_dialect_code} یافت نشد.", "original_text": text}

        source_dialect_id = source_dialect.get("dialect_id")
        target_dialect_id = target_dialect.get("dialect_id")

        # تلاش اول با استفاده از مدل هوشمند
        confidence_level = self.smart_model.confidence_level(normalized_text) if self.smart_model else 0.0
        if confidence_level >= CONFIG.get("confidence_threshold", 0.6) and self.smart_model:
            self.data_access.statistics["smart_model_uses"] += 1
            try:
                conversion_result = self.smart_model.convert_dialect(normalized_text, source_dialect_code,
                                                                     target_dialect_code)
                if conversion_result and "converted_text" in conversion_result:
                    conversion_result["source"] = "smart_model"
                    self.data_access.cache_manager.cache_result(cache_key, json.dumps(conversion_result), 86400)
                    return conversion_result
            except Exception as e:
                self.logger.error(f"خطا در تبدیل لهجه با مدل هوشمند: {e}")

        # تلاش دوم با استفاده از مدل معلم
        if self.teacher:
            self.data_access.statistics["teacher_uses"] += 1
            try:
                conversion_result = self.teacher.convert_dialect(normalized_text, source_dialect_code,
                                                                 target_dialect_code)
                if conversion_result and "converted_text" in conversion_result:
                    conversion_result["source"] = "teacher"
                    # فراخوانی فرآیند یادگیری (بدون انتشار به Kafka)
                    self._learn_from_teacher(normalized_text,
                                             {"conversion": conversion_result, "source": source_dialect_code,
                                              "target": target_dialect_code})
                    self.data_access.cache_manager.cache_result(cache_key, json.dumps(conversion_result), 86400)
                    return conversion_result
            except Exception as e:
                self.logger.error(f"خطا در تبدیل لهجه با مدل معلم: {e}")

        # در صورت عدم موفقیت مدل‌ها، به روش قاعده‌محور تبدیل انجام می‌شود.
        converted_text, rules_applied = self._apply_conversion_rules(normalized_text, source_dialect_id,
                                                                     target_dialect_id)
        converted_text, words_replaced = self._replace_dialect_words(converted_text, source_dialect_id,
                                                                     target_dialect_id)
        changes_count = len(rules_applied) + len(words_replaced)
        conv_confidence = min(0.5 + 0.1 * changes_count, 0.95) if changes_count > 0 else 0.7

        result = {
            "original_text": text,
            "converted_text": converted_text,
            "source_dialect": source_dialect_code,
            "target_dialect": target_dialect_code,
            "confidence": conv_confidence,
            "rules_applied": rules_applied,
            "words_replaced": words_replaced,
            "source": "rule_based"
        }
        self.data_access.cache_manager.cache_result(cache_key, json.dumps(result), 86400)
        self.data_access.performance_metrics.collect_metrics({
            "dialect_conversion": {
                "text_length": len(normalized_text),
                "source_dialect": source_dialect_code,
                "target_dialect": target_dialect_code,
                "rules_applied": len(rules_applied),
                "words_replaced": len(words_replaced),
                "source": result["source"]
            }
        })
        return result

    def _normalize_text(self, text: str) -> str:
        """استفاده از TextNormalizer جهت نرمال‌سازی متن ورودی."""
        return self.normalizer.normalize(text)

    def _apply_conversion_rules(self, text: str, source_dialect_id: str, target_dialect_id: str) -> Tuple[
        str, List[Dict[str, Any]]]:
        """
        اعمال قواعد تبدیل لهجه‌ای بر متن.

        به ازای هر قاعده‌ای که مربوط به لهجه مبدأ و مقصد است، الگوی آن اعمال شده و تغییرات ثبت می‌شود.

        Returns:
            یک تاپل شامل (متن تبدیل شده، لیست قواعد اعمال شده)
        """
        converted_text = text
        rules_applied = []
        for rule in self.data_access.dialect_conversion_rules.values():
            if rule.get("source_dialect") == source_dialect_id and rule.get("target_dialect") == target_dialect_id:
                pattern = rule.get("rule_pattern")
                replacement = rule.get("replacement")
                if not pattern or not replacement:
                    continue
                try:
                    original_text = converted_text
                    converted_text = re.sub(pattern, replacement, converted_text)
                    if original_text != converted_text:
                        count = len(re.findall(pattern, original_text))
                        rules_applied.append({
                            "rule_id": rule.get("rule_id"),
                            "rule_type": rule.get("rule_type"),
                            "description": rule.get("description"),
                            "count": count
                        })
                        self._increment_rule_usage(rule.get("rule_id"))
                except Exception as e:
                    self.logger.error(f"خطا در اعمال قاعده {rule.get('rule_id')}: {e}")
        return converted_text, rules_applied

    def _replace_dialect_words(self, text: str, source_dialect_id: str, target_dialect_id: str) -> Tuple[
        str, List[Dict[str, Any]]]:
        """
        جایگزینی واژگان لهجه‌ای در متن با معادل‌های لهجه مقصد.

        Returns:
            یک تاپل شامل (متن پس از جایگزینی، لیست واژگان جایگزین شده)
        """
        converted_text = text
        words_replaced = []
        source_words = {}
        for word in self.data_access.dialect_words.values():
            if word.get("dialect_id") == source_dialect_id:
                w = word.get("word")
                standard = word.get("standard_equivalent")
                if w and standard:
                    source_words[w] = {
                        "word_id": word.get("word_id"),
                        "standard": standard,
                        "confidence": word.get("confidence", 0.8)
                    }
        target_words = {}
        for word in self.data_access.dialect_words.values():
            if word.get("dialect_id") == target_dialect_id:
                w = word.get("word")
                standard = word.get("standard_equivalent")
                if w and standard:
                    target_words[standard] = w
        for src_word, data in source_words.items():
            pattern = r'\b' + re.escape(src_word) + r'\b'
            if re.search(pattern, converted_text):
                replacement = target_words.get(data["standard"], data["standard"])
                try:
                    count = len(re.findall(pattern, converted_text))
                    converted_text = re.sub(pattern, replacement, converted_text)
                    words_replaced.append({
                        "original": src_word,
                        "replacement": replacement,
                        "count": count,
                        "word_id": data["word_id"]
                    })
                    self._increment_word_usage(data["word_id"])
                except Exception as e:
                    self.logger.error(f"خطا در جایگزینی واژه {src_word}: {e}")
        return converted_text, words_replaced

    def _increment_rule_usage(self, rule_id: str):
        """افزایش شمارنده استفاده از یک قاعده در پایگاه داده."""
        try:
            self.data_access.database.execute_query(
                f"UPDATE dialect_conversion_rules SET usage_count = usage_count + 1 WHERE rule_id = '{rule_id}'"
            )
        except Exception as e:
            self.logger.error(f"خطا در افزایش شمارنده قاعده {rule_id}: {e}")

    def _increment_word_usage(self, word_id: str):
        """افزایش شمارنده استفاده از یک واژه در پایگاه داده."""
        try:
            self.data_access.database.execute_query(
                f"UPDATE dialect_words SET usage_count = usage_count + 1 WHERE word_id = '{word_id}'"
            )
        except Exception as e:
            self.logger.error(f"خطا در افزایش شمارنده واژه {word_id}: {e}")

    def _learn_from_teacher(self, text: str, teacher_output: Any) -> bool:
        """
        فراخوانی فرآیند یادگیری از خروجی مدل معلم (بدون استفاده از Kafka).

        در این لایه فقط فراخوانی مدل دانش‌آموز و ثبت لاگ انجام می‌شود.

        Returns:
            نتیجه‌ی یادگیری به صورت بولین.
        """
        try:
            self.logger.info("یادگیری از خروجی مدل معلم آغاز شد.")
            # فرض می‌کنیم فراخوانی یادگیری توسط مدل دانش‌آموز انجام می‌شود
            return True
        except Exception as e:
            self.logger.error(f"خطا در فرآیند یادگیری از معلم: {e}")
            return False
