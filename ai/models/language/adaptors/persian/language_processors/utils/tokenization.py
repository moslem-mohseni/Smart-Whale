# persian/language_processors/utils/tokenization.py
"""
این ماژول شامل کلاس Tokenizer است که وظیفه‌ی تفکیک جملات و کلمات در متون فارسی را بر عهده دارد.
ابتدا در تلاش است از کتابخانه hazm بهره ببرد؛ در صورت عدم دسترسی، از توابع fallback پیشرفته استفاده می‌کند.
"""

import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from hazm import SentenceTokenizer as HazmSentenceTokenizer, WordTokenizer as HazmWordTokenizer

    USE_HAZM = True
    logger.info("کتابخانه hazm برای توکنیزیشن با موفقیت بارگذاری شد.")
except ImportError:
    USE_HAZM = False
    HazmSentenceTokenizer = None
    HazmWordTokenizer = None
    logger.warning("کتابخانه hazm یافت نشد؛ از توکنایزیشن پیشرفته‌ی داخلی استفاده خواهد شد.")


class Tokenizer:
    def __init__(self):
        """
        مقداردهی اولیه Tokenizer.
        در صورت موجود بودن hazm از ابزارهای آن استفاده می‌شود؛ در غیر این صورت توابع داخلی fallback استفاده می‌گردد.
        """
        if USE_HAZM and HazmSentenceTokenizer and HazmWordTokenizer:
            try:
                self.sentence_tokenizer = HazmSentenceTokenizer()
                self.word_tokenizer = HazmWordTokenizer()
                logger.info("ابزارهای توکنیزیشن hazm با موفقیت راه‌اندازی شدند.")
            except Exception as e:
                logger.error(f"خطا در راه‌اندازی توکنیزیشن hazm: {e}")
                self._initialize_fallback_tokenizers()
        else:
            self._initialize_fallback_tokenizers()

    def _initialize_fallback_tokenizers(self):
        """راه‌اندازی توکنیزیشن پیشرفته داخلی در صورت عدم دسترسی به hazm."""
        self.sentence_tokenizer = self._simple_sentence_tokenizer
        self.word_tokenizer = self._simple_word_tokenizer
        logger.info("ابزارهای توکنیزیشن داخلی (fallback) با موفقیت راه‌اندازی شدند.")

    def tokenize_sentences(self, text: str) -> list:
        """
        تفکیک متن به جملات.

        Args:
            text (str): متن ورودی

        Returns:
            list: لیستی از جملات
        """
        return self.sentence_tokenizer.tokenize(text)

    def tokenize_words(self, text: str) -> list:
        """
        تفکیک متن به کلمات.

        Args:
            text (str): متن ورودی

        Returns:
            list: لیستی از کلمات
        """
        return self.word_tokenizer.tokenize(text)

    @staticmethod
    def _simple_sentence_tokenizer(text: str) -> list:
        """
        تفکیک ساده جملات با استفاده از علائم پایان جمله.

        Args:
            text (str): متن ورودی

        Returns:
            list: لیستی از جملات (بدون علائم پایان)
        """
        # استفاده از الگوی پیشرفته برای تقسیم جملات
        sentences = re.split(r'[.!؟؛]+', text)
        # حذف فضاهای اضافی و جملات خالی
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _simple_word_tokenizer(text: str) -> list:
        """
        تفکیک ساده کلمات با حذف علائم نگارشی و تقسیم بر اساس فاصله.

        Args:
            text (str): متن ورودی

        Returns:
            list: لیستی از کلمات
        """
        # حذف علائم نگارشی (به جز کاراکترهای فارسی)
        cleaned = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        # تقسیم بر اساس فاصله و حذف کلمات خالی
        words = cleaned.split()
        return words


if __name__ == "__main__":
    sample_text = "سلام! امروز هوا خیلی عالی است. آیا شما هم موافقید؟ لطفاً توضیح دهید."
    tokenizer = Tokenizer()
    sentences = tokenizer.tokenize_sentences(sample_text)
    words = tokenizer.tokenize_words(sample_text)

    print("جملات:", sentences)
    print("کلمات:", words)
