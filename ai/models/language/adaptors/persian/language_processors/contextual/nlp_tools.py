# persian/language_processors/utils/nlp_tools.py
"""
ماژول nlp_tools.py

این ماژول ابزارها و توابع پردازش زبان طبیعی برای متون فارسی را فراهم می‌کند.
این ابزارها شامل توابع نرمال‌سازی تکمیلی، توکنیزیشن (تفکیک جملات و کلمات) و سایر
توابع کمکی NLP هستند. در ابتدا تلاش می‌شود تا از کتابخانه hazm بهره برد؛ در صورت عدم
دسترسی، از پیاده‌سازی‌های داخلی پیشرفته (fallback) استفاده می‌شود.

این ماژول به عنوان واسطی عمل می‌کند تا سایر بخش‌های پروژه بتوانند به سادگی از
ابزارهای پردازش متن فارسی بهره ببرند.
"""

import logging
from typing import List

from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# کلاس NLPEngine یک رابط جامع برای ابزارهای NLP فارسی فراهم می‌کند.
class NLPEngine:
    def __init__(self):
        """
        مقداردهی اولیه NLPEngine.
        در این کلاس ابتدا شیء TextNormalizer و Tokenizer ساخته می‌شود.
        اگر hazm موجود باشد، از ابزارهای آن بهره می‌بریم؛ در غیر این صورت نسخه‌های fallback
        پیشرفته و بهینه پیاده‌سازی می‌شوند.
        """
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer()
        logger.info("NLPEngine با موفقیت مقداردهی اولیه شد.")

    def normalize_text(self, text: str) -> str:
        """
        نرمال‌سازی متن ورودی با استفاده از TextNormalizer.

        Args:
            text (str): متن ورودی

        Returns:
            str: متن نرمال‌شده
        """
        try:
            normalized = self.normalizer.normalize(text)
            logger.debug(f"متن اولیه: {text} | متن نرمال‌شده: {normalized}")
            return normalized
        except Exception as e:
            logger.error(f"خطا در نرمال‌سازی متن: {e}")
            raise

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        تفکیک متن به جملات با استفاده از Tokenizer.

        Args:
            text (str): متن ورودی

        Returns:
            List[str]: لیستی از جملات استخراج‌شده
        """
        try:
            sentences = self.tokenizer.tokenize_sentences(text)
            logger.debug(f"تعداد جملات استخراج‌شده: {len(sentences)}")
            return sentences
        except Exception as e:
            logger.error(f"خطا در تفکیک جملات: {e}")
            raise

    def tokenize_words(self, text: str) -> List[str]:
        """
        تفکیک متن به کلمات با استفاده از Tokenizer.

        Args:
            text (str): متن ورودی

        Returns:
            List[str]: لیستی از کلمات استخراج‌شده
        """
        try:
            words = self.tokenizer.tokenize_words(text)
            logger.debug(f"تعداد کلمات استخراج‌شده: {len(words)}")
            return words
        except Exception as e:
            logger.error(f"خطا در تفکیک کلمات: {e}")
            raise

    def process_text(self, text: str) -> dict:
        """
        پردازش جامع متن شامل نرمال‌سازی و توکنیزیشن.
        خروجی شامل متن نرمال‌شده، لیست جملات و لیست کلمات است.

        Args:
            text (str): متن ورودی

        Returns:
            dict: دیکشنری شامل:
                  - normalized: متن نرمال‌شده
                  - sentences: لیست جملات استخراج‌شده
                  - words: لیست کلمات استخراج‌شده
        """
        normalized = self.normalize_text(text)
        sentences = self.tokenize_sentences(normalized)
        words = self.tokenize_words(normalized)
        return {
            "normalized": normalized,
            "sentences": sentences,
            "words": words
        }


if __name__ == "__main__":
    sample_text = (
        "سلام! امروز هوا خیلی عالی است. آیا شما هم موافقید؟ لطفاً توضیح دهید."
    )
    engine = NLPEngine()
    result = engine.process_text(sample_text)
    print("نتیجه پردازش متن:")
    print("متن نرمال‌شده:", result["normalized"])
    print("جملات:", result["sentences"])
    print("کلمات:", result["words"])
