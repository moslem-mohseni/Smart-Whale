from typing import Dict, Any
from .allocator import ResourceAllocator
from .load_balancer import LoadBalancer
from infrastructure.timescaledb.timescale_manager import TimescaleDB

class QuotaManager:
    """
    مدیریت سهمیه‌ی منابع پردازشی برای مدل‌های مختلف جهت جلوگیری از استفاده بیش از حد.
    """

    def __init__(self, default_cpu_quota: float = 20.0, default_memory_quota: int = 2048, default_gpu_quota: float = 10.0):
        """
        مقداردهی اولیه مدیریت سهمیه منابع.
        :param default_cpu_quota: سهمیه‌ی پیش‌فرض CPU برای هر مدل.
        :param default_memory_quota: سهمیه‌ی پیش‌فرض حافظه (MB).
        :param default_gpu_quota: سهمیه‌ی پیش‌فرض GPU.
        """
        self.default_cpu_quota = default_cpu_quota
        self.default_memory_quota = default_memory_quota
        self.default_gpu_quota = default_gpu_quota
        self.quotas: Dict[str, Dict[str, float]] = {}  # ذخیره سهمیه مدل‌ها
        self.allocator = ResourceAllocator()
        self.load_balancer = LoadBalancer()
        self.timescale_db = TimescaleDB()

    def set_quota(self, model_id: str, cpu: float, memory: int, gpu: float = 0.0):
        """
        تنظیم سهمیه‌ی منابع برای یک مدل خاص.
        :param model_id: شناسه مدل.
        :param cpu: حداکثر مقدار CPU که مدل می‌تواند استفاده کند.
        :param memory: حداکثر مقدار حافظه (MB).
        :param gpu: حداکثر مقدار GPU.
        """
        self.quotas[model_id] = {"cpu": cpu, "memory": memory, "gpu": gpu}

        # ثبت در TimescaleDB برای تحلیل سهمیه‌های مصرفی
        self.timescale_db.store_timeseries(
            metric="quota_allocation",
            timestamp=self._get_current_timestamp(),
            tags={"model_id": model_id},
            value={"cpu": cpu, "memory": memory, "gpu": gpu}
        )

    def get_quota(self, model_id: str) -> Dict[str, Any]:
        """
        دریافت سهمیه‌ی تخصیص داده‌شده برای یک مدل خاص.
        :param model_id: شناسه مدل.
        :return: دیکشنری شامل سهمیه‌ی تخصیص‌یافته.
        """
        return self.quotas.get(model_id, {
            "cpu": self.default_cpu_quota,
            "memory": self.default_memory_quota,
            "gpu": self.default_gpu_quota
        })

    def enforce_quota(self, model_id: str, requested_cpu: float, requested_memory: int, requested_gpu: float = 0.0) -> bool:
        """
        بررسی و اعمال محدودیت‌های سهمیه‌ی مدل هنگام تخصیص منابع.
        :param model_id: شناسه مدل.
        :param requested_cpu: مقدار CPU درخواستی.
        :param requested_memory: مقدار حافظه درخواستی (MB).
        :param requested_gpu: مقدار GPU درخواستی.
        :return: `True` اگر تخصیص منابع امکان‌پذیر باشد، `False` اگر محدودیت سهمیه‌ای وجود داشته باشد.
        """
        quota = self.get_quota(model_id)

        if requested_cpu > quota["cpu"] or requested_memory > quota["memory"] or requested_gpu > quota["gpu"]:
            return False  # درخواست بیشتر از سهمیه‌ی مدل است

        # تخصیص منابع از طریق ResourceAllocator
        return self.allocator.allocate(model_id, requested_cpu, requested_memory, requested_gpu)

    def adjust_quota_based_on_usage(self, model_id: str, usage_data: Dict[str, Any]):
        """
        تنظیم سهمیه‌ی مدل بر اساس داده‌های استفاده‌ی قبلی.
        :param model_id: شناسه مدل.
        :param usage_data: داده‌های مربوط به مصرف منابع.
        """
        new_cpu_quota = min(usage_data["cpu"] * 1.2, self.default_cpu_quota * 2)  # افزایش 20٪ در صورت نیاز
        new_memory_quota = min(usage_data["memory"] * 1.2, self.default_memory_quota * 2)
        new_gpu_quota = min(usage_data["gpu"] * 1.2, self.default_gpu_quota * 2)

        self.set_quota(model_id, new_cpu_quota, new_memory_quota, new_gpu_quota)

    def get_all_quotas(self) -> Dict[str, Dict[str, float]]:
        """
        دریافت لیست سهمیه‌های تمامی مدل‌ها.
        :return: دیکشنری شامل سهمیه‌ی تمامی مدل‌ها.
        """
        return self.quotas

    def _get_current_timestamp(self) -> int:
        """
        دریافت زمان جاری به‌صورت `timestamp`.
        """
        import time
        return int(time.time())
