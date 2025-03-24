import psutil

class MemoryAllocator:
    def __init__(self, max_usage_percent=80):
        """
        مدیریت تخصیص حافظه (RAM)
        :param max_usage_percent: حداکثر درصد مجاز استفاده از حافظه (پیش‌فرض: 80٪)
        """
        self.total_memory = psutil.virtual_memory().total  # کل حافظه سیستم (بایت)
        self.max_usage_percent = max_usage_percent  # محدودیت استفاده از حافظه

    def get_memory_status(self):
        """
        دریافت وضعیت فعلی حافظه
        :return: دیکشنری شامل میزان حافظه کل، استفاده‌شده، آزاد و درصد استفاده
        """
        mem = psutil.virtual_memory()
        return {
            "total_memory": mem.total,
            "used_memory": mem.used,
            "available_memory": mem.available,
            "percent_usage": mem.percent
        }

    def can_allocate(self, requested_memory: int) -> bool:
        """
        بررسی امکان تخصیص مقدار مشخصی از حافظه
        :param requested_memory: مقدار حافظه موردنیاز (بایت)
        :return: True اگر تخصیص امکان‌پذیر باشد، False در غیر این‌صورت
        """
        mem = psutil.virtual_memory()
        current_usage_percent = (mem.used + requested_memory) / self.total_memory * 100
        return current_usage_percent <= self.max_usage_percent

    def allocate_memory(self, requested_memory: int):
        """
        تلاش برای تخصیص حافظه در صورت امکان
        :param requested_memory: مقدار حافظه موردنیاز (بایت)
        :return: پیام موفقیت یا خطا
        """
        if self.can_allocate(requested_memory):
            return f"✅ {requested_memory / (1024 ** 2):.2f}MB از حافظه تخصیص یافت."
        else:
            return "❌ تخصیص حافظه ممکن نیست! میزان حافظه موردنیاز بیشتر از حد مجاز است."
