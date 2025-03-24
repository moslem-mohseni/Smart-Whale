# persian/language_processors/utils/regex_utils.py
"""
ماژول regex_utils.py

این ماژول مجموعه‌ای جامع از الگوهای (regex) برای استخراج موجودیت‌های مختلف از متن‌های فارسی را فراهم می‌کند.
این الگوها شامل موارد زیر می‌شوند:
  - ایمیل
  - URL (شامل http و https)
  - شماره تلفن‌های ایرانی (موبایل و ثابت)
  - تاریخ شمسی
  - تاریخ میلادی
  - زمان (با پشتیبانی از فرمت‌های مختلف)
  - مبالغ پولی (با واحدهای رایج)
  - درصد
  - کد ملی ایرانی (10 رقمی)
  - کد پستی ایران (10 رقمی)
  - آدرس IPv4
  - آدرس IPv6
  - هشتگ‌های رسانه‌های اجتماعی (با پشتیبانی از حروف فارسی)
  - منشن‌های شبکه‌های اجتماعی (با پشتیبانی از حروف فارسی)
  - ایموجی (در محدوده‌های رایج)
  - اعداد (به صورت فارسی و انگلیسی)
  - تگ‌های HTML
  - شماره کارت اعتباری (13 تا 16 رقم)
"""

import re
from typing import List, Dict

PATTERNS = {
    "EMAIL": {
        "pattern": r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        "description": "تشخیص آدرس ایمیل"
    },
    "URL": {
        "pattern": r"(https?://[^\s]+)",
        "description": "تشخیص URL شامل http یا https"
    },
    "PHONE_IR": {
        "pattern": r"\b(?:\+98|0098|0)(?:9\d{9}|\d{2,3}-?\d{7,8})\b",
        "description": "تشخیص شماره تلفن ایرانی (موبایل یا تلفن ثابت)"
    },
    "DATE_PERSIAN": {
        "pattern": r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{1,2}\s+("
                   r"?:فروردین|اردیبهشت|خرداد|تیر|مرداد|شهریور|مهر|آبان|آذر|دی|بهمن|اسفند)\s+\d{2,4}\b",
        "description": "تشخیص تاریخ شمسی در چند فرمت"
    },
    "DATE_GREGORIAN": {
        "pattern": r"\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b",
        "description": "تشخیص تاریخ میلادی به صورت YYYY/MM/DD یا YYYY-MM-DD"
    },
    "TIME": {
        "pattern": r"\b\d{1,2}:\d{1,2}(?::\d{1,2})?\s*(?:ق\.ظ|ب\.ظ|AM|PM)?\b",
        "description": "تشخیص الگوی زمان در متن"
    },
    "MONEY": {
        "pattern": r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:تومان|ریال|دلار|یورو|پوند|€|\$|£|¥)\b|\b("
                   r"?:تومان|ریال|دلار|یورو|پوند|€|\$|£|¥)\s?\d+(?:,\d{3})*(?:\.\d+)?\b",
        "description": "تشخیص مبالغ پولی با واحدهای رایج"
    },
    "PERCENT": {
        "pattern": r"\b\d+(?:\.\d+)?\s?(?:٪|%|درصد)\b",
        "description": "تشخیص درصد در متن"
    },
    "NATIONAL_ID": {
        "pattern": r"\b\d{10}\b",
        "description": "تشخیص کد ملی ایرانی (10 رقمی)؛ بررسی صحت عدد به سطح الگوریتمی انجام نمی‌شود"
    },
    "POSTAL_CODE": {
        "pattern": r"\b\d{10}\b",
        "description": "تشخیص کد پستی ایران (10 رقمی)"
    },
    "IPV4": {
        "pattern": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b",
        "description": "تشخیص آدرس IPv4 معتبر"
    },
    "IPV6": {
        "pattern": r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b",
        "description": "تشخیص آدرس IPv6 ساده"
    },
    "HASHTAG": {
        "pattern": r"#([\u0600-\u06FF\w]+)",
        "description": "تشخیص هشتگ‌های رسانه‌های اجتماعی (با پشتیبانی از حروف فارسی)"
    },
    "MENTION": {
        "pattern": r"@([\u0600-\u06FF\w]+)",
        "description": "تشخیص منشن‌های شبکه‌های اجتماعی (با پشتیبانی از حروف فارسی)"
    },
    "EMOJI": {
        # این الگو برخی محدوده‌های رایج ایموجی را پوشش می‌دهد؛ ممکن است نیاز به گسترش داشته باشد.
        "pattern": r"[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]",
        "description": "تشخیص ایموجی‌های رایج در متن (ممکن است تمام ایموجی‌ها را پوشش ندهد)"
    },
    "NUMERIC": {
        "pattern": r"\b[0-9۰-۹]+(?:[.,][0-9۰-۹]+)?\b",
        "description": "تشخیص اعداد به صورت انگلیسی یا فارسی، شامل اعداد کسری"
    },
    "HTML_TAG": {
        "pattern": r"<[^>]+>",
        "description": "تشخیص تگ‌های HTML در متن"
    },
    "CREDIT_CARD": {
        "pattern": r"\b(?:\d[ -]*?){13,16}\b",
        "description": "تشخیص شماره کارت اعتباری (13 تا 16 رقم)"
    }
}


def extract_pattern(text: str, pattern_key: str) -> List[Dict[str, any]]:
    """
    استخراج تمام موارد منطبق با یک الگوی مشخص در متن.

    Args:
        text (str): متن ورودی
        pattern_key (str): کلید الگو در دیکشنری PATTERNS

    Returns:
        List[Dict[str, any]]: فهرستی از دیکشنری‌های شامل {'match': str, 'start': int, 'end': int, 'pattern': str}
    """
    if pattern_key not in PATTERNS:
        return []

    pattern_info = PATTERNS[pattern_key]
    pattern = pattern_info["pattern"]

    matches = []
    try:
        for match_obj in re.finditer(pattern, text):
            matches.append({
                "match": match_obj.group(0),
                "start": match_obj.start(),
                "end": match_obj.end(),
                "pattern": pattern_key
            })
    except re.error as e:
        # در صورت بروز خطا در الگوی regex، خطا را ثبت می‌کند.
        print(f"خطا در اجرای regex برای {pattern_key}: {e}")
    return matches


def extract_all_patterns(text: str, pattern_keys: List[str] = None) -> Dict[str, List[Dict[str, any]]]:
    """
    استخراج تمام موارد منطبق با لیستی از الگوها در متن.
    در صورت عدم ارسال pattern_keys، تمام الگوهای موجود در PATTERNS بررسی می‌شوند.

    Args:
        text (str): متن ورودی
        pattern_keys (List[str], optional): لیستی از کلیدهای الگوهای موردنظر در PATTERNS

    Returns:
        Dict[str, List[Dict[str, any]]]: دیکشنری‌ای که کلید آن نام الگو و مقدار آن لیست نتایج استخراج شده از آن الگو است.
    """
    if pattern_keys is None:
        pattern_keys = list(PATTERNS.keys())

    results = {}
    for key in pattern_keys:
        results[key] = extract_pattern(text, key)
    return results


def remove_repeated_chars(text: str, threshold: int = 3) -> str:
    """
    حذف یا کاهش تکرار بیش از حد کاراکترها (مثلاً "عااااالی" -> "عاالی").
    threshold مشخص می‌کند چند تکرار مجاز باشد.

    Args:
        text (str): متن ورودی
        threshold (int): حداکثر تعداد مجاز تکرار متوالی کاراکتر

    Returns:
        str: متن اصلاح‌شده
    """
    pattern = rf"(.)\1{{{threshold},}}"

    def replacer(match):
        char = match.group(1)
        return char * threshold

    return re.sub(pattern, replacer, text)


def cleanup_text_for_compare(text: str) -> str:
    """
    حذف علائم نگارشی و فاصله‌های اضافی برای مقایسه راحت‌تر متن.
    در سنجش شباهت جملات یا بررسی تکراری بودن متن کاربرد دارد.

    Args:
        text (str): متن ورودی

    Returns:
        str: متن ساده‌سازی‌شده برای مقایسه
    """
    text = re.sub(r"[^\w\s\u0600-\u06FF\d]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


if __name__ == "__main__":
    sample_text = (
        "سلام! آدرس ایمیل من example@test.com هست. "
        "امروز ۱۴۰۱/۰۵/۱۲ و 2021-12-31 هست و ساعت ۱۲:۳۰ ب.ظ است. "
        "شماره تماس من +989123456789 و مبلغ ۲۵۰,۰۰۰ تومان پرداخت شد. "
        "کد ملی من 0043529517 و کد پستی 1234567890 می‌باشد. "
        "آدرس IP من 192.168.1.1 و IPv6 من 2001:0db8:85a3:0000:0000:8a2e:0370:7334 است. "
        "هشتگ‌هایی مانند #سلام و منشن‌هایی مانند @کاربر در متن وجود دارند. "
        "ایموجی‌هایی مانند 😊 و 🚀 نیز ظاهر می‌شوند. "
        "عددهایی مانند 1234.56 و ۱۲۳۴ نیز باید تشخیص داده شوند. "
        "تگ HTML: <div class='sample'>متن</div> و شماره کارت 4111-1111-1111-1111 نیز وجود دارند."
    )

    # تست استخراج یک الگو
    print("==== Test extract_pattern (EMAIL) ====")
    email_matches = extract_pattern(sample_text, "EMAIL")
    print("ایمیل‌ها:", email_matches)

    # تست استخراج همه الگوها
    print("\n==== Test extract_all_patterns ====")
    all_results = extract_all_patterns(sample_text)
    for key, matches in all_results.items():
        print(f"{key}: {matches}")

    # تست حذف کاراکترهای تکراری
    repeated_text = "عااااالیییییی"
    print("\n==== Test remove_repeated_chars ====")
    print("قبل:", repeated_text, "\nبعد:", remove_repeated_chars(repeated_text, threshold=2))

    # تست cleanup_text_for_compare
    compare_text = "این یک متن!! آزمایشی,,, برای مقایسه است."
    print("\n==== Test cleanup_text_for_compare ====")
    print("قبل:", compare_text, "\nبعد:", cleanup_text_for_compare(compare_text))
