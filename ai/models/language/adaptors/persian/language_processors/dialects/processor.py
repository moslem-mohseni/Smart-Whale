# language_processors/dialects/processor.py

"""
ماژول processor.py

این ماژول نقطه ورودی اصلی برای پردازش لهجه‌های فارسی است.
کلاس DialectProcessor عملکردهای تشخیص، تبدیل و یادگیری لهجه را به صورت یکپارچه فراهم می‌کند.
این زیرسیستم بدون وابستگی به Kafka عمل می‌کند.
"""

import json
import logging

from .data_access import DialectDataAccess
from .conversion import DialectConversionProcessor
from .learning import DialectLearningProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DialectProcessor:
    """
    کلاس DialectProcessor نقطه ورودی و هماهنگ‌کننده کل فرآیندهای مربوط به پردازش لهجه است.
    این کلاس شامل متدهایی برای تشخیص لهجه، تبدیل لهجه و فراخوانی فرآیند یادگیری می‌باشد.
    """

    def __init__(self):
        self.logger = logger
        self.data_access = DialectDataAccess()
        self.conversion_processor = DialectConversionProcessor()
        self.learning_processor = DialectLearningProcessor()
        self.logger.info("DialectProcessor initialized successfully.")

    def detect(self, text: str) -> dict:
        """
        تشخیص لهجه متن ورودی.

        Args:
            text (str): متن ورودی

        Returns:
            dict: نتیجه تشخیص لهجه شامل شناسه، نام، کد لهجه، سطح اطمینان و اطلاعات اضافی.
        """
        return self.data_access.detect_dialect(text)

    def convert(self, text: str, target_dialect_code: str = "STANDARD") -> dict:
        """
        تبدیل متن ورودی به لهجه هدف.

        Args:
            text (str): متن ورودی.
            target_dialect_code (str): کد لهجه مقصد (پیش‌فرض: STANDARD).

        Returns:
            dict: نتیجه تبدیل شامل متن تبدیل شده، قوانین اعمال شده، واژگان جایگزین شده، سطح اطمینان و منبع تبدیل.
        """
        return self.conversion_processor.convert_dialect(text, target_dialect_code)

    def learn(self, text: str, teacher_output: dict) -> bool:
        """
        فراخوانی فرآیند یادگیری برای بهبود مدل‌های تشخیص و تبدیل لهجه.

        Args:
            text (str): متن ورودی.
            teacher_output (dict): خروجی آموزشی از مدل معلم.

        Returns:
            bool: نتیجه فرآیند یادگیری (True در صورت موفقیت، False در غیر این صورت).
        """
        return self.learning_processor.learn_from_teacher(text, teacher_output)

    def get_all_dialects(self) -> list:
        """
        دریافت لیست تمام لهجه‌های موجود.

        Returns:
            list: لیست لهجه‌ها به صورت دیکشنری.
        """
        return self.data_access.get_all_dialects()

    def get_statistics(self) -> dict:
        """
        دریافت آمار کلی سیستم لهجه‌بندی.

        Returns:
            dict: شامل آمار دسترسی به داده‌ها، رویدادهای یادگیری و وضعیت سیستم.
        """
        stats = {}
        stats["data_access"] = self.data_access.get_statistics()
        stats["learning"] = self.learning_processor.get_learning_statistics()
        return stats


if __name__ == "__main__":
    processor = DialectProcessor()
    sample_text = "سلام، من میخوام بدونم لهجه من چیست؟"
    detection_result = processor.detect(sample_text)
    print("نتیجه تشخیص لهجه:", json.dumps(detection_result, ensure_ascii=False, indent=2))

    conversion_result = processor.convert(sample_text, target_dialect_code="TEHRANI")
    print("نتیجه تبدیل لهجه:", json.dumps(conversion_result, ensure_ascii=False, indent=2))

    # فرض خروجی مدل معلم برای یادگیری:
    sample_teacher_output = {"info": "مثال خروجی آموزشی برای بهبود مدل"}
    learn_result = processor.learn(sample_text, sample_teacher_output)
    print("نتیجه یادگیری:", learn_result)

    stats = processor.get_statistics()
    print("آمار کلی سیستم:", json.dumps(stats, ensure_ascii=False, indent=2))
