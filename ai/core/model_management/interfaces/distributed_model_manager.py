from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio


class ModelType(Enum):
    """انواع مدل‌های قابل پشتیبانی"""
    LANGUAGE = "language"  # مدل‌های پردازش زبان
    VISION = "vision"  # مدل‌های پردازش تصویر
    AUDIO = "audio"  # مدل‌های پردازش صوت
    MULTIMODAL = "multimodal"  # مدل‌های چندوجهی
    CUSTOM = "custom"  # مدل‌های سفارشی


@dataclass
class ModelMetrics:
    """متریک‌های عملکردی مدل"""
    latency: float  # تأخیر پردازش (میلی‌ثانیه)
    throughput: float  # توان عملیاتی (درخواست بر ثانیه)
    error_rate: float  # نرخ خطا
    memory_usage: float  # میزان مصرف حافظه (مگابایت)
    gpu_usage: Optional[float]  # میزان مصرف GPU (درصد)
    last_updated: datetime  # آخرین به‌روزرسانی


@dataclass
class ModelInstance:
    """نمونه اجرایی یک مدل"""
    instance_id: str  # شناسه یکتای نمونه
    model_type: ModelType  # نوع مدل
    host: str  # آدرس میزبان
    port: int  # پورت سرویس
    status: str  # وضعیت فعلی
    metrics: ModelMetrics  # متریک‌های عملکردی
    last_heartbeat: datetime  # آخرین اعلام وضعیت
    capabilities: List[str]  # قابلیت‌های این نمونه
    max_batch_size: int  # حداکثر اندازه دسته برای پردازش


class DistributedModelManager(ABC):
    """
    مدیر توزیع‌شده مدل‌ها

    این کلاس مسئول مدیریت و هماهنگی نمونه‌های مختلف مدل‌ها در یک محیط توزیع‌شده است.
    قابلیت‌هایی مانند تعادل بار، مدیریت خطا و مقیاس‌پذیری خودکار را فراهم می‌کند.
    """

    @abstractmethod
    async def register_model(self, instance: ModelInstance) -> bool:
        """
        ثبت یک نمونه جدید از مدل در سیستم توزیع‌شده

        Args:
            instance: مشخصات نمونه مدل جدید

        Returns:
            موفقیت‌آمیز بودن ثبت
        """
        pass

    @abstractmethod
    async def unregister_model(self, instance_id: str) -> bool:
        """
        حذف یک نمونه مدل از سیستم

        Args:
            instance_id: شناسه نمونه مدل

        Returns:
            موفقیت‌آمیز بودن حذف
        """
        pass

    @abstractmethod
    async def get_available_instances(self, model_type: ModelType) -> List[ModelInstance]:
        """
        دریافت لیست نمونه‌های فعال و در دسترس یک نوع مدل

        Args:
            model_type: نوع مدل مورد نظر

        Returns:
            لیست نمونه‌های در دسترس
        """
        pass

    @abstractmethod
    async def select_instance(self, model_type: ModelType,
                              requirements: Optional[Dict[str, Any]] = None) -> ModelInstance:
        """
        انتخاب بهترین نمونه برای اجرای یک درخواست

        Args:
            model_type: نوع مدل مورد نیاز
            requirements: نیازمندی‌های خاص (مثل حداقل سرعت، حداکثر تأخیر و غیره)

        Returns:
            مناسب‌ترین نمونه برای اجرای درخواست
        """
        pass

    @abstractmethod
    async def update_metrics(self, instance_id: str, metrics: ModelMetrics) -> None:
        """
        به‌روزرسانی متریک‌های یک نمونه

        Args:
            instance_id: شناسه نمونه
            metrics: متریک‌های جدید
        """
        pass

    @abstractmethod
    async def handle_failure(self, instance_id: str, error: Exception) -> None:
        """
        مدیریت خطای یک نمونه

        Args:
            instance_id: شناسه نمونه
            error: خطای رخ داده
        """
        pass

    @abstractmethod
    async def scale_models(self, model_type: ModelType,
                           metrics: Dict[str, float]) -> None:
        """
        مقیاس‌پذیری خودکار برای یک نوع مدل

        Args:
            model_type: نوع مدل
            metrics: متریک‌های فعلی سیستم
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        بررسی سلامت کل سیستم توزیع‌شده

        Returns:
            وضعیت سلامت سیستم
        """
        pass

    @abstractmethod
    async def backup_state(self) -> None:
        """پشتیبان‌گیری از وضعیت فعلی سیستم"""
        pass

    @abstractmethod
    async def restore_state(self) -> None:
        """بازیابی وضعیت سیستم از پشتیبان"""
        pass

    async def __aenter__(self):
        """پشتیبانی از context manager"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """پاکسازی منابع هنگام خروج"""
        await self.backup_state()