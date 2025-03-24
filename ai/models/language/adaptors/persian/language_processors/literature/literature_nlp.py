# persian/language_processors/literature/literature_nlp.py
"""
ماژول literature_nlp.py

این ماژول شامل توابعی برای پیش‌پردازش متون ادبی فارسی است.
امکانات اصلی:
  - نرمال‌سازی متون ادبی با حفظ ساختارهای خاص (مثلاً حفظ خطوط جدید در شعر)
  - توکنیزیشن و استخراج کلمات کلیدی با استفاده از توکنایزر و فیلتر کردن کلمات توقف
  - محاسبه شباهت متنی بین دو متن ادبی با استفاده از SequenceMatcher
"""

import re
import logging
from difflib import SequenceMatcher
from typing import List

from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# لیست کلمات توقف (stopwords) نمونه برای متون ادبی فارسی
STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "با", "را", "های", "برای",
    "است", "تا", "هم", "او", "یا", "اما", "باید", "می", "آن", "شود",
    "هر", "نیز", "ها", "بر", "بود", "شد", "یک", "خود", "ما", "کرد", "شده"
}


def normalize_literature(text: str) -> str:
    """
    نرمال‌سازی متون ادبی فارسی با حفظ ساختارهای ویژه مانند خطوط جدید و پاراگراف‌بندی.

    Args:
        text (str): متن ورودی ادبی.

    Returns:
        str: متن نرمال‌شده.
    """
    try:
        normalizer = TextNormalizer()
        normalized = normalizer.normalize(text)
        # حفظ ساختارهای خطی؛ کاهش فضای اضافی بین خطوط
        normalized = re.sub(r'\n+', '\n', normalized)
        return normalized.strip()
    except Exception as e:
        logger.error(f"خطا در نرمال‌سازی متن ادبی: {e}")
        return text


def extract_keywords_literature(text: str) -> List[str]:
    """
    استخراج کلمات کلیدی از متون ادبی.
    از Tokenizer برای تقسیم‌بندی متن به کلمات استفاده می‌شود و سپس کلمات توقف و کلمات کوتاه حذف می‌گردند.

    Args:
        text (str): متن ادبی.

    Returns:
        List[str]: لیست کلمات کلیدی استخراج‌شده.
    """
    try:
        tokenizer = Tokenizer()
        words = tokenizer.tokenize_words(text)
    except Exception as e:
        logger.error(f"خطا در توکنیزیشن متن ادبی: {e}")
        words = text.split()

    # فیلتر کردن کلمات توقف و کلمات با طول کمتر از 3 حرف
    keywords = [word for word in words if word not in STOPWORDS and len(word) > 2]

    # حفظ یکتایی کلمات به ترتیب اولیه
    seen = set()
    unique_keywords = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            unique_keywords.append(word)

    logger.debug(f"کلمات کلیدی استخراج‌شده: {unique_keywords}")
    return unique_keywords


def calculate_text_similarity_literature(text1: str, text2: str) -> float:
    """
    محاسبه شباهت متنی بین دو متن ادبی با استفاده از SequenceMatcher.

    Args:
        text1 (str): متن اول.
        text2 (str): متن دوم.

    Returns:
        float: میزان شباهت (بین 0 تا 1).
    """
    try:
        similarity = SequenceMatcher(None, text1, text2).ratio()
        logger.debug(f"شباهت بین '{text1}' و '{text2}' برابر {similarity:.2f} است.")
        return similarity
    except Exception as e:
        logger.error(f"خطا در محاسبه شباهت متن: {e}")
        return 0.0


if __name__ == "__main__":
    sample_text = """هر آنکه آفتاب عشق در دل دارد،
از تاریکی شب بی‌خبر است.
در میان پرده‌های ظلمت،
نور امید تابان می‌شود."""

    norm_text = normalize_literature(sample_text)
    print("متن نرمال‌شده:", norm_text)

    keywords = extract_keywords_literature(norm_text)
    print("کلمات کلیدی استخراج‌شده:", keywords)

    sim = calculate_text_similarity_literature("نور امید", "نور امید تابان")
    print("شباهت متنی:", sim)
