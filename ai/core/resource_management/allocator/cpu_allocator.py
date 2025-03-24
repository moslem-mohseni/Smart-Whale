import os
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class CPUAllocator:
    def __init__(self, max_workers=None):
        """
        مدیریت تخصیص پردازنده (CPU)
        :param max_workers: حداکثر تعداد پردازش‌های همزمان (پیش‌فرض: نصف تعداد هسته‌های CPU)
        """
        self.total_cores = os.cpu_count() or 1  # تعداد کل هسته‌های پردازنده
        self.max_workers = max_workers if max_workers else max(1, self.total_cores // 2)
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def get_cpu_usage(self):
        """
        دریافت میزان استفاده فعلی از پردازنده (درصد)
        :return: مقدار استفاده از CPU به‌صورت درصدی
        """
        return psutil.cpu_percent(interval=1)

    def allocate_task(self, function, *args, use_threads=False):
        """
        تخصیص پردازنده برای اجرای یک وظیفه
        :param function: تابعی که باید اجرا شود
        :param args: آرگومان‌های تابع
        :param use_threads: اگر True باشد از ThreadPoolExecutor استفاده می‌شود (مناسب برای I/O)
        :return: Future object برای مدیریت اجرای وظیفه
        """
        executor = ThreadPoolExecutor(max_workers=self.max_workers) if use_threads else self.executor
        return executor.submit(function, *args)

    def shutdown(self):
        """ آزادسازی منابع اختصاص‌یافته به پردازنده """
        self.executor.shutdown(wait=True)
