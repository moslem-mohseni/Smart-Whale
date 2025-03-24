# persian/language_processors/grammar/pos_analyzer.py
"""
ماژول pos_analyzer.py

این ماژول شامل کلاس POSAnalyzer است که وظیفه استخراج و تحلیل برچسب‌های دستوری (POS) متون فارسی را بر عهده دارد.
این کلاس با استفاده از ابزارهای نرمال‌سازی (TextNormalizer) و توکنیزیشن (Tokenizer) از پوشه utils،
همچنین کتابخانه hazm (در صورت موجودیت) برچسب‌های POS را استخراج می‌کند.
در صورت عدم دسترسی به hazm، این کلاس به صورت fallback عمل کرده و خروجی خالی برمی‌گرداند.
علاوه بر استخراج توالی POS، قابلیت‌هایی جهت بررسی اعتبار توالی برچسب‌ها نیز فراهم شده است.
"""

import os
import re
import logging
from typing import List, Tuple, Dict, Any, Optional

from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class POSAnalyzer:
    """
    کلاس POSAnalyzer وظیفه استخراج و تحلیل برچسب‌های دستوری (POS) متون فارسی را بر عهده دارد.

    امکانات اصلی:
      - بارگذاری POS Tagger از کتابخانه hazm با پشتیبانی از مسیرهای مختلف مدل.
      - استخراج برچسب‌های POS برای متن نرمال‌شده.
      - تولید توالی POS به‌عنوان رشته.
      - بررسی اعتبار توالی POS با استفاده از الگوهای از پیش تعریف‌شده.

    نکته: امکان تزریق وابستگی برای TextNormalizer و Tokenizer جهت تست و سفارشی‌سازی فراهم شده است.
    """

    def __init__(self, language: str = "persian",
                 normalizer: TextNormalizer = None,
                 tokenizer: Tokenizer = None):
        self.language = language
        self.logger = logger
        self.normalizer = normalizer if normalizer is not None else TextNormalizer()
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.pos_tagger = None  # POS Tagger در اینجا بارگذاری می‌شود

        # تلاش برای بارگذاری POS Tagger از hazm با چند مسیر ممکن
        try:
            from hazm import POSTagger
        except ImportError:
            self.logger.warning("کتابخانه hazm یافت نشد؛ تحلیل POS ممکن است محدود باشد.")
            return

        pos_model_paths = [
            "adaptors/persian/language_processor/postagger.model",  # مسیر اصلی
            os.path.join(os.path.dirname(__file__), "../../models/postagger.model"),  # مسیر نسبی
            "models/postagger.model"  # مسیر ساده
        ]
        for path in pos_model_paths:
            try:
                self.pos_tagger = POSTagger(model=path)
                self.logger.info(f"POS Tagger از مسیر {path} با موفقیت بارگذاری شد.")
                break
            except Exception as e:
                self.logger.debug(f"مسیر {path} ناموفق: {e}")
        if self.pos_tagger is None:
            self.logger.warning("POS Tagger بارگذاری نشد؛ تحلیل POS غیر فعال خواهد بود.")

    def analyze(self, text: str) -> List[Tuple[str, str]]:
        """
        تحلیل برچسب‌های دستوری متن ورودی.

        Args:
            text (str): متن ورودی (خام یا نرمال‌شده)

        Returns:
            List[Tuple[str, str]]: لیستی از توپل‌های (کلمه، برچسب POS)
        """
        normalized_text = self.normalizer.normalize(text)
        if self.pos_tagger is None:
            self.logger.warning("POS Tagger در دسترس نیست؛ خروجی تحلیل POS خالی خواهد بود.")
            return []
        try:
            words = self.tokenizer.tokenize_words(normalized_text)
            pos_tags = self.pos_tagger.tag(words)
            self.logger.debug(f"برچسب‌های POS استخراج شده: {pos_tags}")
            return pos_tags
        except Exception as e:
            self.logger.error(f"خطا در تحلیل POS: {e}")
            return []

    def get_pos_sequence(self, text: str) -> str:
        """
        استخراج توالی برچسب‌های POS به صورت رشته.

        Args:
            text (str): متن ورودی

        Returns:
            str: توالی برچسب‌ها به صورت یک رشته (مثلاً "N V ADJ")
        """
        pos_tags = self.analyze(text)
        sequence = " ".join(tag for _, tag in pos_tags)
        return sequence

    def validate_pos_sequence(self, pos_sequence: str) -> Dict[str, Any]:
        """
        بررسی اعتبار توالی POS بر اساس الگوهای از پیش تعریف‌شده.
        در این نسخه به صورت نمونه، تنها بررسی می‌شود که آیا توالی طولانی‌تر از ۲ کلمه است یا خیر.

        Args:
            pos_sequence (str): توالی برچسب‌های POS

        Returns:
            Dict[str, Any]: دیکشنری شامل:
                - is_valid (bool): وضعیت اعتبار توالی
                - details (str): توضیحاتی در مورد اعتبار توالی
        """
        if len(pos_sequence.split()) >= 2:
            return {"is_valid": True, "details": "توالی POS معتبر است."}
        else:
            return {"is_valid": False, "details": "توالی POS کوتاه است."}

    def check_pos_pattern_validity(self, text: str) -> Dict[str, Any]:
        """
        ترکیبی از استخراج توالی POS و بررسی اعتبار آن.

        Args:
            text (str): متن ورودی

        Returns:
            Dict[str, Any]: شامل توالی POS، وضعیت اعتبار و توضیحات
        """
        pos_sequence = self.get_pos_sequence(text)
        validity = self.validate_pos_sequence(pos_sequence)
        return {
            "pos_sequence": pos_sequence,
            "validity": validity
        }


if __name__ == "__main__":
    sample_text = "سلام، علی امروز به مدرسه رفت و خیلی خوشحال بود."
    analyzer = POSAnalyzer()
    pos_tags = analyzer.analyze(sample_text)
    pos_seq = analyzer.get_pos_sequence(sample_text)
    validity_info = analyzer.check_pos_pattern_validity(sample_text)

    print("برچسب‌های POS:", pos_tags)
    print("توالی POS:", pos_seq)
    print("اعتبار توالی:", validity_info)
