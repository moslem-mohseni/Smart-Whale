from enum import Enum

class BackoffStrategy(Enum):
    """
    استراتژی‌های Backoff برای کنترل تأخیر بین تلاش‌های مجدد
    """
    CONSTANT = "constant"       # تأخیر ثابت (مثلاً همیشه 1 ثانیه)
    EXPONENTIAL = "exponential" # تأخیر نمایی (1s → 2s → 4s → 8s ...)
    RANDOMIZED = "randomized"   # تأخیر تصادفی (بین 0 تا مقدار پایه × شماره تلاش)
