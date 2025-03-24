import time
import random
import logging
from ai.core.resilience.retry.backoff_strategies import BackoffStrategy

class RetryManager:
    def __init__(self, max_attempts=3, backoff_strategy=BackoffStrategy.CONSTANT, base_delay=1):
        """
        مدیریت تلاش مجدد برای عملیات‌های ناموفق
        :param max_attempts: حداکثر تعداد تلاش‌های مجدد (پیش‌فرض: 3)
        :param backoff_strategy: استراتژی Backoff برای تأخیر بین تلاش‌ها
        :param base_delay: تأخیر اولیه برای تلاش مجدد (ثانیه)
        """
        self.max_attempts = max_attempts
        self.backoff_strategy = backoff_strategy
        self.base_delay = base_delay
        self.logger = logging.getLogger("RetryManager")

    def execute_with_retry(self, function, *args, **kwargs):
        """
        اجرای یک تابع با پشتیبانی از تلاش مجدد در صورت شکست
        :param function: تابعی که باید اجرا شود
        :param args: آرگومان‌های تابع
        :param kwargs: آرگومان‌های کلیدی تابع
        :return: نتیجه اجرای موفقیت‌آمیز تابع یا None در صورت شکست
        """
        for attempt in range(1, self.max_attempts + 1):
            try:
                self.logger.info(f"🚀 تلاش {attempt} برای اجرای تابع {function.__name__}...")
                result = function(*args, **kwargs)
                self.logger.info(f"✅ عملیات {function.__name__} در تلاش {attempt} موفق بود!")
                return result
            except Exception as e:
                self.logger.warning(f"⚠️ تلاش {attempt} برای {function.__name__} ناموفق بود: {e}")

                if attempt < self.max_attempts:
                    delay = self.calculate_backoff_delay(attempt)
                    self.logger.info(f"⏳ انتظار {delay:.2f} ثانیه قبل از تلاش بعدی...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"❌ عملیات {function.__name__} پس از {self.max_attempts} تلاش شکست خورد.")
                    return None

    def calculate_backoff_delay(self, attempt):
        """
        محاسبه مقدار تأخیر بر اساس استراتژی Backoff
        :param attempt: شماره تلاش فعلی
        :return: مقدار تأخیر بر حسب ثانیه
        """
        if self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            return self.base_delay * (2 ** (attempt - 1))  # 1s → 2s → 4s ...
        elif self.backoff_strategy == BackoffStrategy.RANDOMIZED:
            return random.uniform(0, self.base_delay * attempt)  # مقدار تصادفی
        else:  # Backoff ثابت
            return self.base_delay
