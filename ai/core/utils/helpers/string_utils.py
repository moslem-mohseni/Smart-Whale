import re
from datetime import datetime

class StringUtils:
    @staticmethod
    def to_lowercase(s: str) -> str:
        """ تبدیل رشته به حروف کوچک """
        return s.lower() if s else ""

    @staticmethod
    def to_uppercase(s: str) -> str:
        """ تبدیل رشته به حروف بزرگ """
        return s.upper() if s else ""

    @staticmethod
    def trim(s: str) -> str:
        """ حذف فاصله‌های اضافی از ابتدا و انتهای رشته """
        return s.strip() if s else ""

    @staticmethod
    def contains(s: str, substring: str) -> bool:
        """ بررسی وجود یک زیررشته در رشته اصلی """
        return substring in s if s else False

    @staticmethod
    def format_date(date_str: str, input_format: str, output_format: str) -> str:
        """ تبدیل فرمت تاریخ از یک قالب به قالب دیگر """
        try:
            date_obj = datetime.strptime(date_str, input_format)
            return date_obj.strftime(output_format)
        except ValueError:
            return f"❌ فرمت تاریخ نامعتبر: {date_str}"
