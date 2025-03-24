import logging
import re


class LogProcessor(logging.Filter):
    SENSITIVE_PATTERNS = [
        (r"\b\d{16}\b", "[REDACTED_CREDIT_CARD]"),  # شماره کارت بانکی
        (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),  # شماره تأمین اجتماعی
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "[REDACTED_EMAIL]")  # ایمیل
    ]

    def __init__(self, min_log_level=logging.INFO, extra_fields=None):
        """
        پردازش و فیلتر کردن لاگ‌ها
        :param min_log_level: حداقل سطح لاگ برای پردازش
        :param extra_fields: فیلدهای اضافی برای اضافه کردن به لاگ‌ها
        """
        super().__init__()
        self.min_log_level = min_log_level
        self.extra_fields = extra_fields or {}

    def filter(self, record):
        """ فیلتر کردن لاگ‌ها بر اساس سطح و حذف داده‌های حساس """
        if record.levelno < self.min_log_level:
            return False  # حذف لاگ‌هایی که پایین‌تر از سطح تنظیم‌شده هستند

        record.msg = self.mask_sensitive_data(record.getMessage())

        # اضافه کردن فیلدهای اضافی به لاگ
        for key, value in self.extra_fields.items():
            setattr(record, key, value)

        return True

    def mask_sensitive_data(self, message):
        """ حذف داده‌های حساس از پیام لاگ """
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            message = re.sub(pattern, replacement, message)
        return message
