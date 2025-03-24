# persian/language_processors/grammar/rule_manager.py

"""
ماژول rule_manager.py

این ماژول مسئول مدیریت قواعد گرامری در زبان فارسی است.
کلاس RuleManager وظیفه بارگذاری، ذخیره، افزودن و به‌روزرسانی قواعد گرامری را بر عهده دارد.
همچنین امکان تحلیل متن به‌وسیله‌ی اعمال قواعد موجود و تولید لیست خطاها به همراه پیشنهادات اصلاح را فراهم می‌کند.
در صورتی که هیچ قانونی در پایگاه داده موجود نباشد، قواعد پیش‌فرض به‌طور خودکار اضافه می‌شوند.
"""

import json
import logging
import re
import time
import uuid
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional

# واردات تنظیمات و زیرساخت‌ها
from ...config import CONFIG
from ..grammar.pos_analyzer import POSAnalyzer  # در صورت نیاز به تحلیل POS (اختیاری)
from ..utils.misc import compute_statistics  # به عنوان مثال برای محاسبات آماری

# فرض بر این است که ماژول‌های زیرساختی مانند RedisAdapter، CacheManager، ClickHouseDB و غیره در دسترس هستند.
from ai.models.language.infrastructure.caching.cache_manager import CacheManager
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _generate_correction_template(pattern: str, suggested: str) -> str:
    """
    ایجاد قالب اصلاح (correction template) برای قاعده گرامری.
    در این پیاده‌سازی ساده، پیشنهاد اصلاح به عنوان قالب اصلاح استفاده می‌شود.
    """
    return suggested


class RuleManager:
    """
    RuleManager

    این کلاس مسئول مدیریت قواعد گرامری در زبان فارسی است.
    وظایف اصلی شامل:
      - بارگذاری قواعد گرامری از پایگاه داده (و یا بارگذاری قواعد پیش‌فرض در صورت عدم وجود)
      - ذخیره و به‌روزرسانی قواعد گرامری
      - تحلیل متن بر اساس قواعد گرامری و استخراج خطاها به همراه پیشنهادات اصلاح
      - کشف قواعد جدید بر مبنای خطاهای یافت شده
      - ارائه متدهایی برای آمارگیری، ورود/خروجی قواعد
    """

    def __init__(self):
        self.logger = logger
        self.database = ClickHouseDB()
        self.cache_manager = CacheManager()
        # دیکشنری محلی برای نگهداری قواعد گرامری؛ کلید: rule_id، مقدار: دیکشنری اطلاعات قاعده
        self.grammar_rules: Dict[str, Dict[str, Any]] = self._load_rules()
        # (اختیاری) تحلیل POS – می‌تواند برای ارزیابی ساختار نیز استفاده شود
        self.pos_analyzer = POSAnalyzer()

        # آمار عملکرد داخلی
        self.statistics = {
            "load_time": time.time(),
            "requests": 0,
            "cache_hits": 0,
            "new_rules_discovered": 0,
            "rules_total": len(self.grammar_rules)
        }

        self.logger.info(f"RuleManager راه‌اندازی شد. تعداد قواعد بارگذاری‌شده: {len(self.grammar_rules)}")

    def _load_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری قواعد گرامری از پایگاه داده.
        در صورت عدم وجود، قواعد پیش‌فرض را مقداردهی اولیه می‌کند.
        """
        try:
            result = self.database.execute_query("SELECT * FROM grammar_rules")
            if result and len(result) > 0:
                rules = {row['rule_id']: row for row in result}
                self.logger.info(f"تعداد قواعد گرامری بازیابی‌شده از پایگاه داده: {len(rules)}")
                return rules
            else:
                self.logger.info("هیچ قاعده‌ای در پایگاه داده یافت نشد؛ بارگذاری قواعد پیش‌فرض.")
                self._initialize_default_rules()
                # بازیابی مجدد پس از افزودن قواعد پیش‌فرض
                result = self.database.execute_query("SELECT * FROM grammar_rules")
                return {row['rule_id']: row for row in result} if result else {}
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری قواعد گرامری: {e}")
            return {}

    def _initialize_default_rules(self):
        """
        مقداردهی اولیه قواعد گرامری پیش‌فرض در صورت عدم وجود قواعد در پایگاه داده.
        """
        default_rules = [
            {
                "rule_id": "space_before_punctuation",
                "rule_name": "فاصله قبل از علائم نگارشی",
                "rule_type": "punctuation",
                "pattern": r"\s+([،؛:.!؟])",
                "correction": r"\1",
                "examples": ["سلام , چطوری ؟", "خوبم . ممنون !"],
                "confidence": 0.95,
                "source": "default",
                "usage_count": 1,
                "discovery_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "rule_id": "missing_space_after_punctuation",
                "rule_name": "فاصله بعد از علائم نگارشی",
                "rule_type": "punctuation",
                "pattern": r"([،؛:.!؟])(\w)",
                "correction": r"\1 \2",
                "examples": ["سلام,چطوری؟خوبم.", "چه خبر?هیچی"],
                "confidence": 0.90,
                "source": "default",
                "usage_count": 1,
                "discovery_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "rule_id": "missing_space_verb",
                "rule_name": "فاصله بین اجزای فعل",
                "rule_type": "spacing",
                "pattern": r"(می|نمی)(\w+)",
                "correction": r"\1\u200c\2",  # استفاده از Zero-width non-joiner
                "examples": ["میروم", "نمیدانم", "میخواهم"],
                "confidence": 0.85,
                "source": "default",
                "usage_count": 1,
                "discovery_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "rule_id": "extra_space_verb",
                "rule_name": "فاصله اضافی در فعل",
                "rule_type": "spacing",
                "pattern": r"(می|نمی) (\w+)",
                "correction": r"\1\u200c\2",
                "examples": ["می روم", "نمی دانم", "می خواهم"],
                "confidence": 0.85,
                "source": "default",
                "usage_count": 1,
                "discovery_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "rule_id": "ha_plural",
                "rule_name": "جمع با «ها»",
                "rule_type": "spacing",
                "pattern": r"(\w+)ها",
                "correction": r"\1‌ها",
                "examples": ["کتابها", "ماشینها", "خانهها"],
                "confidence": 0.80,
                "source": "default",
                "usage_count": 1,
                "discovery_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "rule_id": "tar_comparative",
                "rule_name": "صفت تفضیلی با «تر»",
                "rule_type": "spacing",
                "pattern": r"(\w+)تر",
                "correction": r"\1‌تر",
                "examples": ["بزرگتر", "بهتر", "زیباتر"],
                "confidence": 0.75,
                "source": "default",
                "usage_count": 1,
                "discovery_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "rule_id": "tarin_superlative",
                "rule_name": "صفت عالی با «ترین»",
                "rule_type": "spacing",
                "pattern": r"(\w+)ترین",
                "correction": r"\1‌ترین",
                "examples": ["بزرگترین", "بهترین", "زیباترین"],
                "confidence": 0.75,
                "source": "default",
                "usage_count": 1,
                "discovery_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "rule_id": "repeated_words",
                "rule_name": "تکرار کلمات",
                "rule_type": "redundancy",
                "pattern": r"\b(\w+)[ ]+\1\b",
                "correction": r"\1",
                "examples": ["سلام سلام", "من من", "این این"],
                "confidence": 0.70,
                "source": "default",
                "usage_count": 1,
                "discovery_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        ]

        for rule in default_rules:
            self.store_rule(rule)
        self.logger.info(f"قواعد پیش‌فرض گرامری مقداردهی اولیه شدند. تعداد: {len(default_rules)}")

    def store_rule(self, rule: Dict[str, Any]):
        """
        ذخیره یا به‌روزرسانی یک قاعده گرامری در پایگاه داده.
        """
        try:
            existing = self.database.execute_query(
                f"SELECT rule_id FROM grammar_rules WHERE rule_id='{rule['rule_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("grammar_rules", rule)
                self.logger.debug(f"قاعده جدید ذخیره شد: {rule['rule_id']}")
            else:
                self.database.execute_query(f"""
                    UPDATE grammar_rules
                    SET usage_count = usage_count + 1
                    WHERE rule_id = '{rule['rule_id']}'
                """)
                self.logger.debug(f"قاعده موجود به‌روزرسانی شد: {rule['rule_id']}")
        except Exception as e:
            self.logger.error(f"خطا در ذخیره قاعده گرامری {rule.get('rule_id')}: {e}")

    def add_rule(self, rule: Dict[str, Any]) -> bool:
        """
        افزودن یک قاعده گرامری جدید به سیستم.
        """
        try:
            self.store_rule(rule)
            self.grammar_rules[rule['rule_id']] = rule
            self.statistics["new_rules_discovered"] += 1
            self.logger.info(f"قاعده گرامری جدید افزوده شد: {rule['rule_id']}")
            return True
        except Exception as e:
            self.logger.error(f"خطا در افزودن قاعده گرامری: {e}")
            return False

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """
        دریافت تمامی قواعد گرامری موجود.
        """
        return list(self.grammar_rules.values())

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """
        تحلیل متن ورودی بر اساس قواعد گرامری و استخراج خطاها همراه با پیشنهادات اصلاح.
        """
        self.statistics["requests"] += 1
        errors = []
        for rule_id, rule in self.grammar_rules.items():
            pattern = rule.get("pattern", "")
            if not pattern:
                continue
            try:
                for match in re.finditer(pattern, text):
                    matched_text = match.group(0)
                    start_pos = match.start()
                    suggested = re.sub(pattern, rule.get("correction", ""), matched_text)
                    if matched_text != suggested:
                        errors.append({
                            "word": matched_text,
                            "position": start_pos,
                            "suggested": suggested,
                            "error_type": rule.get("rule_type", "Unknown"),
                            "rule_id": rule_id
                        })
            except Exception as e:
                self.logger.error(f"خطا در تحلیل با قاعده {rule_id}: {e}")
                continue
        return errors

    def discover_new_rules(self, text: str, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        کشف قواعد جدید بر اساس خطاهای یافت‌شده در متن.
        """
        new_rules = []
        for error in errors:
            if "rule_id" in error and error["rule_id"] in self.grammar_rules:
                continue
            word = error.get("word", "")
            suggested = error.get("suggested", "")
            error_type = error.get("error_type", "Unknown")
            if not word or not suggested or word == suggested:
                continue
            try:
                pattern_str = self._generate_pattern_from_error(word, suggested)
                if not pattern_str:
                    continue
                rule_id = str(uuid.uuid4())
                similar = self.find_similar_rule(pattern_str)
                if similar:
                    # به‌روزرسانی قواعد مشابه
                    similar["examples"].append(word)
                    self.database.execute_query(f"""
                        UPDATE grammar_rules
                        SET confidence = LEAST(confidence + 0.05, 1.0),
                            usage_count = usage_count + 1
                        WHERE rule_id = '{similar['rule_id']}'
                    """)
                    continue
                new_rule = {
                    "rule_id": rule_id,
                    "rule_name": f"Rule Auto: {error_type}",
                    "rule_type": error_type,
                    "pattern": pattern_str,
                    "correction": _generate_correction_template(pattern_str, suggested),
                    "examples": [word],
                    "confidence": 0.6,
                    "source": "auto_discovery",
                    "usage_count": 1,
                    "discovery_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                new_rules.append(new_rule)
            except Exception as e:
                self.logger.error(f"خطا در کشف قاعده جدید: {e}")
                continue
        return new_rules

    def _generate_pattern_from_error(self, word: str, suggested: str) -> str:
        """
        ایجاد الگوی regex از خطای گرامری بر مبنای تفاوت بین کلمه اشتباه و پیشنهاد اصلاح.
        """
        try:
            matcher = SequenceMatcher(None, word, suggested)
            if matcher.ratio() > 0.6:
                pattern = re.escape(word)
                # تعمیم الگو برای علائم نگارشی (در صورت وجود)
                if re.search(r'[،؛:!?.؟]', word):
                    pattern = pattern.replace('\\،', '[،؛:!?.؟]').replace('\\؛', '[،؛:!?.؟]')
                    pattern = pattern.replace('\\:', '[،؛:!?.؟]').replace('\\!', '[،؛:!?.؟]')
                    pattern = pattern.replace('\\?', '[،؛:!?.؟]')
                return pattern
            return ""
        except Exception as e:
            self.logger.error(f"خطا در ایجاد الگوی regex: {e}")
            return ""

    def find_similar_rule(self, pattern: str) -> Optional[Dict[str, Any]]:
        """
        یافتن قاعده‌ای مشابه با الگوی داده‌شده بر مبنای شباهت regex.
        """
        for rule in self.grammar_rules.values():
            rule_pattern = rule.get("pattern", "")
            if not rule_pattern:
                continue
            similarity = SequenceMatcher(None, pattern, rule_pattern).ratio()
            if similarity > 0.8:
                return rule
        return None

    def update_rules(self) -> List[Dict[str, Any]]:
        """
        به‌روزرسانی قواعد گرامری از پایگاه داده و تازه‌سازی دیکشنری داخلی.
        Returns:
            لیست قواعد جدید پس از به‌روزرسانی
        """
        self.grammar_rules = self._load_rules()
        self.statistics["rules_total"] = len(self.grammar_rules)
        self.logger.info(f"قواعد گرامری به‌روزرسانی شدند. تعداد جدید: {len(self.grammar_rules)}")
        return list(self.grammar_rules.values())

    def get_statistics(self) -> Dict[str, Any]:
        """
        دریافت آمار عملکرد داخلی RuleManager.
        Returns:
            دیکشنری حاوی آمار مانند load_time، requests، ...
        """
        return dict(self.statistics)

    def export_rules(self, filename: str) -> str:
        """
        خروجی گرفتن از قواعد در قالب فایل JSON.
        """
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(list(self.grammar_rules.values()), f, ensure_ascii=False, indent=2)
            self.logger.info(f"Rules exported to {filename}")
            return f"Rules exported to {filename}"
        except Exception as e:
            self.logger.error(f"Error exporting rules to {filename}: {e}")
            return "Error exporting rules"

    def import_rules(self, filename: str) -> str:
        """
        وارد کردن قواعد از فایل JSON.
        """
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for rule in data:
                    self.store_rule(rule)
            self.logger.info(f"Rules imported from {filename}")
            return f"Rules imported from {filename}"
        except Exception as e:
            self.logger.error(f"Error importing rules from {filename}: {e}")
            return "Error importing rules"
