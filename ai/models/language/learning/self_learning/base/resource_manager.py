"""
ResourceManager Module
-------------------------
این فایل مسئول مدیریت منابع سیستم خودآموزی (Self-Learning) است.
این کلاس به عنوان مدیر منابع عمل می‌کند تا تعداد درخواست‌های همزمان (مثلاً وظایف آموزش یا پردازش داده) را کنترل کند.
از مکانیزم Semaphore ناهمزمان برای تخصیص منابع استفاده شده و متریک‌های مربوط به مصرف منابع ثبت می‌شوند.
این نسخه نهایی و عملیاتی است که با بهترین و هوشمندترین مکانیسم‌ها و کارایی بالا پیاده‌سازی شده است.
"""

import asyncio
import logging
from typing import Optional

from .base_component import BaseComponent


class ResourceManager(BaseComponent):
    """
    ResourceManager مسئول مدیریت منابع محاسباتی و کنترل تعداد وظایف همزمان در سیستم خودآموزی است.

    ویژگی‌ها:
      - استفاده از یک Semaphore برای محدودسازی تعداد وظایف همزمان.
      - امکان درخواست (allocate) و آزادسازی (release) منابع به صورت ناهمزمان.
      - ثبت متریک‌های مربوط به تخصیص منابع (مثلاً تعداد درخواست‌های موفق و در انتظار).
      - ارائه یک context manager ناهمزمان برای تخصیص و آزادسازی منابع به صورت خودکار.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        مقداردهی اولیه ResourceManager.

        Args:
            config (Optional[dict]): پیکربندی اختصاصی که می‌تواند شامل حداکثر تعداد وظایف همزمان (max_concurrent_tasks) باشد.
        """
        super().__init__(component_type="resource_manager", config=config)
        max_tasks = self.config.get("max_concurrent_tasks", 5)
        self.semaphore = asyncio.Semaphore(max_tasks)
        self.max_concurrent_tasks = max_tasks
        self.logger.info(f"[ResourceManager] Initialized with max_concurrent_tasks={max_tasks}")

    async def allocate_resource(self) -> None:
        """
        درخواست تخصیص یک واحد از منابع (کاهش شمارنده Semaphore).
        این متد به صورت ناهمزمان اجرا می‌شود.
        """
        self.logger.debug("[ResourceManager] Waiting to allocate resource...")
        await self.semaphore.acquire()
        self.logger.debug(
            f"[ResourceManager] Resource allocated. Available: {self.semaphore._value}/{self.max_concurrent_tasks}")
        self.increment_metric("resource_allocated")

    def release_resource(self) -> None:
        """
        آزادسازی یک واحد از منابع (افزایش شمارنده Semaphore).
        """
        self.semaphore.release()
        self.logger.debug(
            f"[ResourceManager] Resource released. Available: {self.semaphore._value}/{self.max_concurrent_tasks}")
        self.increment_metric("resource_released")

    async def allocate_resource_context(self):
        """
        یک context manager ناهمزمان جهت تخصیص و آزادسازی خودکار منابع.
        Usage:
            async with resource_manager.allocate_resource_context():
                # کارهایی که نیاز به منابع دارند
        """
        await self.allocate_resource()
        try:
            yield
        finally:
            self.release_resource()

    def get_status(self) -> dict:
        """
        دریافت وضعیت فعلی منابع.

        Returns:
            dict: شامل تعداد منابع آزاد و کل تعداد منابع.
        """
        status = {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "available_resources": self.semaphore._value  # NOTE: _value attribute is internal
        }
        self.logger.debug(f"[ResourceManager] Current status: {status}")
        return status


# نمونه تستی برای ResourceManager
if __name__ == "__main__":
    async def dummy_task(task_id: int, resource_manager: ResourceManager):
        await resource_manager.allocate_resource()
        try:
            resource_manager.logger.info(f"Task {task_id} started using resource.")
            # شبیه‌سازی کار: به مدت 1 ثانیه
            await asyncio.sleep(1)
            resource_manager.logger.info(f"Task {task_id} completed.")
        finally:
            resource_manager.release_resource()


    async def main():
        rm = ResourceManager(config={"max_concurrent_tasks": 3})
        tasks = [asyncio.create_task(dummy_task(i, rm)) for i in range(6)]
        await asyncio.gather(*tasks)
        print("Final Resource Status:", rm.get_status())


    asyncio.run(main())
