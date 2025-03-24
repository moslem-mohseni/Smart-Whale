import re

class SensitiveDataFilter:
    DEFAULT_PATTERNS = {
        "credit_card": (r"\b\d{16}\b", "[REDACTED_CREDIT_CARD]"),  # شماره کارت بانکی
        "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),  # شماره تأمین اجتماعی
        "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "[REDACTED_EMAIL]")  # ایمیل
    }

    def __init__(self, custom_patterns=None):
        """
        فیلتر کردن داده‌های حساس از لاگ‌ها
        :param custom_patterns: دیکشنری شامل الگوهای سفارشی برای ماسک کردن داده‌های حساس
        """
        self.patterns = self.DEFAULT_PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)

    def mask_sensitive_data(self, message):
        """ جایگزین کردن داده‌های حساس در پیام لاگ """
        for _, (pattern, replacement) in self.patterns.items():
            message = re.sub(pattern, replacement, message)
        return message
