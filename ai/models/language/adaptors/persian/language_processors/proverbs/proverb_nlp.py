# persian/language_processors/proverbs/proverb_nlp.py
"""
ماژول proverb_nlp.py

این ماژول شامل توابعی برای پردازش زبان ضرب‌المثل‌های فارسی می‌باشد.
امکانات اصلی:
  - نرمال‌سازی ضرب‌المثل‌ها با استفاده از TextNormalizer.
  - استخراج کلمات کلیدی از ضرب‌المثل (با استفاده از توکنایزر و فیلتر کردن کلمات غیرضروری).
  - محاسبه شباهت متنی بین دو عبارت.
این فایل وظیفه آماده‌سازی داده‌ها برای سایر بخش‌های زیرسیستم ضرب‌المثل را بر عهده دارد.
"""

import re
import logging
from difflib import SequenceMatcher
from typing import List

from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# تعریف لیست کلمات توقف (Stopwords) به‌صورت نمونه
STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "با", "را", "های", "برای",
    "است", "تا", "هم", "او", "یا", "اما", "باید", "می", "آن", "شود",
    "هر", "نیز", "ها", "بر", "بود", "شد", "یک", "خود", "ما", "کرد",
    "شده", "اگر", "چه", "کنید", "وی", "همه", "کند"
}


def normalize_proverb(text: str) -> str:
    """
    نرمال‌سازی ضرب‌المثل با استفاده از TextNormalizer.

    Args:
        text (str): متن ضرب‌المثل

    Returns:
        str: متن نرمال‌شده
    """
    try:
        normalizer = TextNormalizer()
        normalized = normalizer.normalize(text)
        logger.debug(f"متن نرمال‌شده: {normalized}")
        return normalized
    except Exception as e:
        logger.error(f"خطا در نرمال‌سازی ضرب‌المثل: {e}")
        return text


def extract_keywords(text: str) -> List[str]:
    """
    استخراج کلمات کلیدی از ضرب‌المثل.
    از توکنایزر برای تقسیم‌بندی استفاده می‌شود و سپس کلمات توقف و کلمات کوتاه فیلتر می‌شوند.

    Args:
        text (str): متن ضرب‌المثل

    Returns:
        List[str]: لیست کلمات کلیدی استخراج‌شده
    """
    try:
        tokenizer = Tokenizer()
        # توکنایز کردن متن به کلمات
        words = tokenizer.tokenize_words(text)
    except Exception as e:
        logger.error(f"خطا در توکنیزیشن برای استخراج کلمات کلیدی: {e}")
        words = text.split()

    # فیلتر کردن کلمات توقف و کلمات با طول کمتر از 3 حرف
    keywords = [word for word in words if word not in STOPWORDS and len(word) > 2]
    # حذف تکراری‌ها
    unique_keywords = list(dict.fromkeys(keywords))
    logger.debug(f"کلمات کلیدی استخراج‌شده: {unique_keywords}")
    return unique_keywords


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    محاسبه شباهت متنی بین دو عبارت با استفاده از SequenceMatcher.

    Args:
        text1 (str): متن اول
        text2 (str): متن دوم

    Returns:
        float: میزان شباهت (بین 0 تا 1)
    """
    try:
        similarity = SequenceMatcher(None, text1, text2).ratio()
        logger.debug(f"شباهت بین '{text1}' و '{text2}' برابر {similarity:.2f} است.")
        return similarity
    except Exception as e:
        logger.error(f"خطا در محاسبه شباهت: {e}")
        return 0.0


if __name__ == "__main__":
    sample_proverb = "هر که بامش بیش برفش بیشتر"
    normalized = normalize_proverb(sample_proverb)
    print("متن نرمال‌شده:", normalized)
    keywords = extract_keywords(normalized)
    print("کلمات کلیدی:", keywords)
    similarity = calculate_text_similarity("هر که بامش بیش", "هر که بامش بیش برفش بیشتر")
    print("شباهت:", similarity)
