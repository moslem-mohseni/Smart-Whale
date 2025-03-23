"""
Training ResourceManager Module
--------------------------------
این فایل مسئول مدیریت منابع اختصاصی مربوط به فرآیند آموزش (Training) در سیستم خودآموزی است.
این ماژول به صورت اختصاصی برای تخصیص منابع آموزشی (مانند GPU memory slots، فضای حافظه برای داده‌های دسته‌ای و ...)
طراحی شده و از مکانیسم‌های ناهمزمان (asyncio.Semaphore) جهت کنترل تعداد وظایف همزمان آموزش بهره می‌برد.

این نسخه نهایی و عملیاتی با بهترین و هوشمندترین مکانیسم‌ها برای مدیریت منابع آموزشی پیاده‌سازی شده است.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from ..base.base_component import BaseComponent


class TrainingResourceManager(BaseComponent):
    """
    TrainingResourceManager مسئول مدیریت منابع اختصاصی برای فرآیندهای آموزشی است.

    امکانات:
      - استفاده از Semaphore ناهمزمان جهت محدود کردن تعداد وظایف آموزش همزمان.
      - فراهم کردن context manager ناهمزمان جهت تخصیص و آزادسازی منابع به صورت خودکار.
      - ثبت و گزارش متریک‌های مرتبط با تخصیص و آزادسازی منابع آموزشی.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        راه‌اندازی اولیه TrainingResourceManager.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری که می‌تواند شامل:
                - "max_training_slots": حداکثر تعداد وظایف آموزشی همزمان (پیش‌فرض: 4)
                - سایر تنظیمات مرتبط با مدیریت منابع آموزشی.
        """
        super().__init__(component_type="training_resource_manager", config=config)
        self.logger = logging.getLogger("TrainingResourceManager")
        max_slots = int(self.config.get("max_training_slots", 4))
        self.semaphore = asyncio.Semaphore(max_slots)
        self.max_training_slots = max_slots
        self.logger.info(f"[TrainingResourceManager] Initialized with max_training_slots={max_slots}")

    async def allocate_resource(self) -> None:
        """
        درخواست تخصیص یک واحد از منابع آموزشی.
        این متد به صورت ناهمزمان اجرا می‌شود و منتظر آزاد شدن یک training slot می‌ماند.
        """
        self.logger.debug("[TrainingResourceManager] Waiting to allocate training resource...")
        await self.semaphore.acquire()
        self.logger.debug(
            f"[TrainingResourceManager] Allocated training resource. Available: {self.semaphore._value}/{self.max_training_slots}")
        self.increment_metric("training_resource_allocated")

    def release_resource(self) -> None:
        """
        آزادسازی یک واحد از منابع آموزشی.
        """
        self.semaphore.release()
        self.logger.debug(
            f"[TrainingResourceManager] Released training resource. Available: {self.semaphore._value}/{self.max_training_slots}")
        self.increment_metric("training_resource_released")

    async def allocate_resource_context(self):
        """
        یک context manager ناهمزمان جهت تخصیص و آزادسازی منابع آموزشی به صورت خودکار.
        استفاده:
            async with training_resource_manager.allocate_resource_context():
                # اجرای وظیفه آموزشی
        """
        await self.allocate_resource()
        try:
            yield
        finally:
            self.release_resource()

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی تخصیص منابع آموزشی.

        Returns:
            Dict[str, Any]: شامل تعداد اسلات‌های در دسترس و کل تعداد اسلات‌ها.
        """
        status = {
            "max_training_slots": self.max_training_slots,
            "available_slots": self.semaphore._value  # NOTE: _value به عنوان یک مقدار داخلی استفاده می‌شود.
        }
        self.logger.debug(f"[TrainingResourceManager] Current status: {status}")
        return status


# نمونه تستی برای TrainingResourceManager (برای تست در حالت توسعه؛ در محیط تولید به کار رود)
if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.DEBUG)


    async def dummy_training_task(task_id: int, manager: TrainingResourceManager):
        await manager.allocate_resource()
        try:
            manager.logger.info(f"Training task {task_id} started using allocated resource.")
            await asyncio.sleep(1)  # شبیه‌سازی فرآیند آموزشی
            manager.logger.info(f"Training task {task_id} completed.")
        finally:
            manager.release_resource()


    async def main():
        trm = TrainingResourceManager(config={"max_training_slots": 3})
        tasks = [asyncio.create_task(dummy_training_task(i, trm)) for i in range(6)]
        await asyncio.gather(*tasks)
        print("Final training resource status:", trm.get_status())


    asyncio.run(main())
