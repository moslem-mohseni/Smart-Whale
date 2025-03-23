# ai/core/common/resource_config.py

"""
تنظیمات مربوط به منابع سیستم
این فایل پارامترهای محدودیت مصرف منابع را تعریف می‌کند
"""

from dataclasses import dataclass
import torch


@dataclass
class ResourceConfig:
    """تنظیمات منابع سیستم"""

    # تنظیمات CPU
    max_cpu_threads: int = 12  # 75% از کل هسته‌ها
    cpu_threshold: float = 80.0  # حداکثر درصد استفاده از CPU

    # تنظیمات حافظه
    max_memory_gb: float = 24.0  # 75% از کل RAM
    memory_threshold: float = 85.0  # حداکثر درصد استفاده از حافظه

    # تنظیمات GPU
    cuda_available: bool = torch.cuda.is_available()
    gpu_memory_fraction: float = 0.8  # استفاده از 80% حافظه GPU
    gpu_threshold: float = 85.0  # حداکثر درصد استفاده از GPU

    # تنظیمات بچ‌سایز
    default_batch_size: int = 32
    max_batch_size: int = 128

    # تنظیمات کش
    model_cache_size_gb: float = 4.0
    data_cache_size_gb: float = 8.0

    def __post_init__(self):
        if self.cuda_available:
            self.device = torch.device('cuda')
            # تنظیم حافظه GPU
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
        else:
            self.device = torch.device('cpu')