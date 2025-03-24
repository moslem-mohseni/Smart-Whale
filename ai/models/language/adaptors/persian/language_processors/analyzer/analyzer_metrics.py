# language_processors/analyzer/analyzer_metrics.py

"""
ماژول analyzer_metrics.py

این فایل مسئول جمع‌آوری و گزارش متریک‌های عملکرد زیرسیستم آنالیز است.
کلاس AnalyzerMetrics وظیفه ثبت تعداد درخواست‌های پردازش، زمان‌های پاسخ، نرخ موفقیت و
سایر آمارهای مهم را بر عهده دارد.

می‌توان این متریک‌ها را برای مانیتورینگ و بهینه‌سازی عملکرد سیستم استفاده نمود.
"""

import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnalyzerMetrics:
    """
    کلاس AnalyzerMetrics مسئول ثبت و گزارش متریک‌های عملکرد آنالیز است.
    """

    def __init__(self):
        self.logger = logger
        self.reset_metrics()

    def reset_metrics(self) -> None:
        """
        بازنشانی تمامی متریک‌های جمع‌آوری‌شده.
        """
        self.metrics = {
            "requests_processed": 0,
            "total_processing_time": 0.0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "start_time": time.time()
        }
        self.logger.info("متریک‌های آنالیز بازنشانی شدند.")

    def record_request(self, processing_time: float, success: bool, cache_hit: bool = False) -> None:
        """
        ثبت یک درخواست پردازش شده به همراه زمان پردازش و وضعیت موفقیت آن.

        Args:
            processing_time (float): زمان پردازش درخواست (بر حسب ثانیه)
            success (bool): True در صورت موفقیت، False در صورت خطا
            cache_hit (bool, optional): True اگر نتیجه از کش بازیابی شده، False در غیر این صورت (پیش‌فرض: False)
        """
        self.metrics["requests_processed"] += 1
        self.metrics["total_processing_time"] += processing_time
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

        self.logger.debug(
            f"درخواست ثبت شد: زمان پردازش={processing_time}, موفق={success}, کش_hit={cache_hit}"
        )

    def get_average_processing_time(self) -> float:
        """
        محاسبه میانگین زمان پردازش درخواست‌ها.

        Returns:
            float: میانگین زمان پردازش در صورت وجود درخواست، در غیر این صورت 0.0
        """
        total = self.metrics["requests_processed"]
        if total == 0:
            return 0.0
        return self.metrics["total_processing_time"] / total

    def get_success_rate(self) -> float:
        """
        محاسبه نرخ موفقیت درخواست‌ها.

        Returns:
            float: نسبت درخواست‌های موفق به کل درخواست‌ها (بین 0 تا 1)
        """
        total = self.metrics["requests_processed"]
        if total == 0:
            return 0.0
        return self.metrics["successful_requests"] / total

    def get_cache_hit_rate(self) -> float:
        """
        محاسبه نرخ موفقیت کش.

        Returns:
            float: نسبت cache hits به کل دسترسی‌های کش (بین 0 تا 1)
        """
        total_cache = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_cache == 0:
            return 0.0
        return self.metrics["cache_hits"] / total_cache

    def get_uptime(self) -> float:
        """
        محاسبه زمان فعالیت سیستم (از زمان شروع تا اکنون).

        Returns:
            float: زمان فعالیت بر حسب ثانیه.
        """
        return time.time() - self.metrics["start_time"]

    def get_metrics(self) -> Dict[str, float]:
        """
        بازگرداندن تمامی متریک‌های جمع‌آوری شده به صورت دیکشنری.

        Returns:
            Dict[str, float]: شامل تعداد درخواست‌ها، میانگین زمان پردازش، نرخ موفقیت، نرخ کش و زمان فعالیت.
        """
        metrics_summary = {
            "requests_processed": self.metrics["requests_processed"],
            "average_processing_time": self.get_average_processing_time(),
            "success_rate": self.get_success_rate(),
            "cache_hit_rate": self.get_cache_hit_rate(),
            "uptime_seconds": self.get_uptime()
        }
        return metrics_summary
