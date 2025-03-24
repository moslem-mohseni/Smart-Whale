# persian/language_processors/utils/misc.py

import logging
import re
import statistics
from datetime import datetime
from typing import Any, List, Optional, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# تلاش برای وارد کردن کتابخانه jdatetime برای کار با تاریخ شمسی
try:
    import jdatetime

    USE_JDATETIME = True
    logger.info("کتابخانه jdatetime با موفقیت بارگذاری شد.")
except ImportError:
    USE_JDATETIME = False
    logger.warning("کتابخانه jdatetime یافت نشد؛ از نسخه fallback استفاده خواهد شد.")


    class DummyJDateTime:
        def __init__(self, year: int, month: int, day: int):
            self.year = year
            self.month = month
            self.day = day

        @classmethod
        def strptime(cls, date_string: str, fmt: str):
            """
            به صورت ساده تاریخ شمسی را بر اساس فرمت ورودی پردازش می‌کند.
            فرض می‌کنیم تاریخ ورودی به صورت شمسی است.
            """
            try:
                dt = datetime.strptime(date_string, fmt)
                # فرض می‌کنیم ورودی شمسی است؛ برای تبدیل تقریبی، سال میلادی برابر است با سال شمسی + 621
                return cls(dt.year, dt.month, dt.day)
            except Exception as e:
                logger.error(f"خطا در تبدیل رشته تاریخ به DummyJDateTime: {e}")
                raise

        def togregorian(self) -> datetime:
            """
            تبدیل تقریبی تاریخ شمسی به تاریخ میلادی (سال میلادی = سال شمسی + 621)
            """
            try:
                g_year = self.year + 621
                return datetime(g_year, self.month, self.day)
            except Exception as e:
                logger.error(f"خطا در تبدیل DummyJDateTime به میلادی: {e}")
                raise

        @classmethod
        def fromgregorian(cls, dt: datetime):
            """
            تبدیل تقریبی تاریخ میلادی به تاریخ شمسی (سال شمسی = سال میلادی - 621)
            """
            try:
                j_year = dt.year - 621
                return cls(j_year, dt.month, dt.day)
            except Exception as e:
                logger.error(f"خطا در تبدیل تاریخ میلادی به DummyJDateTime: {e}")
                raise

        def strftime(self, fmt: str) -> str:
            """
            فرمت‌دهی تاریخ با استفاده از تاریخ میلادی تقریبی.
            """
            try:
                return self.togregorian().strftime(fmt)
            except Exception as e:
                logger.error(f"خطا در فرمت‌دهی DummyJDateTime: {e}")
                raise


    # ایجاد یک فضای نامی ساده برای jdatetime
    class DummyJdatetimeModule:
        datetime = DummyJDateTime


    jdatetime = DummyJdatetimeModule()


def jalali_to_gregorian(jalali_date: str, fmt: str = "%Y/%m/%d") -> Optional[datetime]:
    """
    تبدیل تاریخ شمسی به تاریخ میلادی.
    """
    try:
        j_date = jdatetime.datetime.strptime(jalali_date, fmt)
        g_date = j_date.togregorian()
        return g_date
    except Exception as e:
        logger.error(f"خطا در تبدیل تاریخ شمسی به میلادی: {e}")
        return None


def gregorian_to_jalali(greg_date: datetime, fmt: str = "%Y/%m/%d") -> Optional[str]:
    """
    تبدیل تاریخ میلادی به تاریخ شمسی.
    """
    try:
        # استفاده از آرگومان موقعیتی
        j_date = jdatetime.datetime.fromgregorian(greg_date)
        return j_date.strftime(fmt)
    except Exception as e:
        logger.error(f"خطا در تبدیل تاریخ میلادی به شمسی: {e}")
        return None


def compute_statistics(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "variance": None, "stdev": None}
    try:
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        variance_val = statistics.variance(values) if len(values) > 1 else 0.0
        stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
        return {
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "variance": round(variance_val, 4),
            "stdev": round(stdev_val, 4)
        }
    except Exception as e:
        logger.error(f"خطا در محاسبه آمار: {e}")
        return {"mean": None, "median": None, "variance": None, "stdev": None}


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        return dt.strftime(fmt)
    except Exception as e:
        logger.error(f"خطا در فرمت‌دهی تاریخ: {e}")
        return ""


def parse_json(json_string: str) -> Optional[Dict[str, Any]]:
    try:
        import json
        return json.loads(json_string)
    except Exception as e:
        logger.error(f"خطا در پارس کردن JSON: {e}")
        return None


if __name__ == "__main__":
    jalali = "1401/05/12"
    greg = jalali_to_gregorian(jalali)
    logger.info(f"تاریخ میلادی برای {jalali}: {greg}")

    if greg:
        jal = gregorian_to_jalali(greg)
        logger.info(f"تبدیل مجدد میلادی به شمسی: {jal}")

    sample_values = [1, 2, 3, 4, 5, 6]
    stats = compute_statistics(sample_values)
    logger.info(f"آمار نمونه: {stats}")

    now = datetime.now()
    logger.info(f"تاریخ فرمت‌شده: {format_datetime(now)}")

    json_str = '{"key": "value", "number": 123}'
    parsed = parse_json(json_str)
    logger.info(f"نتیجه پارس JSON: {parsed}")
