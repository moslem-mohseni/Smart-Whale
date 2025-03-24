# persian/language_processors/grammar/correction_engine.py

"""
ماژول correction_engine.py

این ماژول به عنوان زیرسیستم اصلاح گرامر متن‌های فارسی عمل می‌کند.
کلاس CorrectionEngine وظیفه اصلاح خودکار خطاهای گرامری را بر عهده دارد.
این فایل ورودی (متن فارسی) را دریافت کرده، از طریق ماژول‌های RuleManager و POSAnalyzer تحلیل گرامری انجام می‌دهد،
و سپس اصلاحات استخراج‌شده را به صورت نزولی (از انتها به ابتدا) روی متن اعمال می‌کند.
خروجی شامل متن اصلاح‌شده، لیستی از اصلاحات اعمال‌شده و زمان پردازش است.
"""

import json
import time
import logging
from typing import Dict, Any, List

from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer
from .rule_manager import RuleManager
from .pos_analyzer import POSAnalyzer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CorrectionEngine:
    """
    CorrectionEngine

    این کلاس وظیفه اصلاح خودکار خطاهای گرامری متن‌های فارسی را بر عهده دارد.
    فرآیند اصلاح به این صورت است:
      1. نرمال‌سازی متن ورودی.
      2. تحلیل گرامری متن با استفاده از RuleManager (مدیریت قواعد گرامری پیشرفته).
      3. تحلیل ساختار دستوری متن با استفاده از POSAnalyzer (اختیاری برای بهبود دقت اصلاح).
      4. اعمال اصلاحات استخراج‌شده به صورت نزولی (به منظور حفظ ایندکس‌ها).
      5. بازگرداندن متن اصلاح‌شده همراه با جزئیات اصلاحات و زمان پردازش.

    نکته: استفاده از الگوی dependency injection برای اجزای زیرسیستم (مانند RuleManager و POSAnalyzer) امکان تست و جایگزینی ساده‌تر را فراهم می‌کند.
    """

    def __init__(self,
                 normalizer: TextNormalizer = None,
                 tokenizer: Tokenizer = None,
                 rule_manager: RuleManager = None,
                 pos_analyzer: POSAnalyzer = None):
        self.logger = logger
        # استفاده از اجزای تزریق‌شده در صورت وجود؛ در غیر این صورت از پیاده‌سازی پیش‌فرض استفاده می‌شود.
        self.normalizer = normalizer if normalizer is not None else TextNormalizer()
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.rule_manager = rule_manager if rule_manager is not None else RuleManager()
        self.pos_analyzer = pos_analyzer if pos_analyzer is not None else POSAnalyzer()
        self.logger.info("CorrectionEngine با موفقیت مقداردهی اولیه شد.")

    def correct_text(self, text: str) -> Dict[str, Any]:
        """
        اصلاح خودکار خطاهای گرامری متن ورودی.

        Args:
            text (str): متن ورودی فارسی.

        Returns:
            Dict[str, Any]: دیکشنری شامل:
                - corrected_text: متن نهایی پس از اعمال اصلاحات.
                - corrections: لیستی از اصلاحات اعمال‌شده (شامل کلمه اصلی، پیشنهاد، موقعیت و نوع خطا).
                - processing_time: زمان مصرف‌شده برای اصلاح (به ثانیه).
        """
        start_time = time.time()

        # گام ۱: نرمال‌سازی متن
        try:
            normalized_text = self.normalizer.normalize(text)
        except Exception as e:
            self.logger.error(f"خطا در نرمال‌سازی متن: {e}")
            normalized_text = text

        self.logger.debug(f"متن نرمال‌شده: {normalized_text}")

        # گام ۲: تحلیل گرامری جهت استخراج خطاها و پیشنهادات اصلاح
        try:
            grammar_errors = self.rule_manager.analyze(normalized_text)
        except Exception as e:
            self.logger.error(f"خطا در تحلیل گرامری: {e}")
            grammar_errors = []

        self.logger.info(f"تعداد خطاهای گرامری یافت‌شده: {len(grammar_errors)}")

        # گام ۳: (اختیاری) تحلیل POS برای بررسی ساختار دستوری
        try:
            pos_tags = self.pos_analyzer.analyze(normalized_text)
            self.logger.debug(f"نتایج تحلیل POS: {pos_tags}")
        except Exception as e:
            self.logger.error(f"خطا در تحلیل ساختار دستوری (POS): {e}")
            pos_tags = []

        # گام ۴: اعمال اصلاحات به صورت نزولی جهت جلوگیری از تغییر نادرست ایندکس‌ها
        corrected_text = normalized_text
        corrections_applied: List[Dict[str, Any]] = []
        sorted_errors = sorted(grammar_errors, key=lambda err: err.get("position", 0), reverse=True)

        for error in sorted_errors:
            original_word = error.get("word", "")
            suggestion = error.get("suggested", "")
            pos = error.get("position", -1)
            if original_word and suggestion and pos >= 0:
                # بررسی تطابق کلمه در موقعیت تعیین‌شده
                segment = corrected_text[pos: pos + len(original_word)]
                if segment != original_word:
                    self.logger.warning(
                        f"عدم تطابق در موقعیت {pos}: انتظار '{original_word}'، دریافت '{segment}'. اصلاح رد شد."
                    )
                    continue
                # اعمال اصلاح: تقسیم متن به سه بخش و جایگزینی بخش موردنظر
                before = corrected_text[:pos]
                after = corrected_text[pos + len(original_word):]
                corrected_text = before + suggestion + after
                corrections_applied.append({
                    "original": original_word,
                    "suggested": suggestion,
                    "position": pos,
                    "error_type": error.get("error_type", "Unknown")
                })
                self.logger.debug(
                    f"اصلاح '{original_word}' به '{suggestion}' در موقعیت {pos} انجام شد."
                )

        processing_time = time.time() - start_time
        result = {
            "corrected_text": corrected_text,
            "corrections": corrections_applied,
            "processing_time": round(processing_time, 4)
        }
        self.logger.info(
            f"اصلاح متن به پایان رسید. تعداد اصلاحات: {len(corrections_applied)} | زمان پردازش: {result['processing_time']} ثانیه."
        )
        return result


if __name__ == "__main__":
    sample_text = (
        "این یک متن تستی است که شامل اشتباهاتی مانند میروم به مدرسه, کتابها و سلام سلام می‌باشد."
    )
    engine = CorrectionEngine()
    result = engine.correct_text(sample_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
