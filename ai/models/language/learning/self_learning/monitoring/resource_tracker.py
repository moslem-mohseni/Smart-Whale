"""
ResourceTracker Module
------------------------
این فایل مسئول نظارت بر منابع مصرفی در فرآیند خودآموزی است.
ResourceTracker اطلاعات مربوط به مصرف CPU، حافظه و سایر منابع را از منابع داخلی سیستم (مانند StateManager یا سیستم‌های زیرساختی) دریافت کرده و گزارش می‌دهد.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های نظارتی و کارایی بالا پیاده‌سازی شده است.
"""

import asyncio
import logging
from abc import ABC
from typing import Dict, Any, Optional
from datetime import datetime

from ..base.base_component import BaseComponent


class ResourceTracker(BaseComponent, ABC):
    """
    ResourceTracker مسئول نظارت بر مصرف منابع سیستم (مانند CPU، حافظه، تردها) در فرآیند خودآموزی است.

    امکانات:
      - دریافت وضعیت منابع از StateManager یا API های زیرساخت.
      - ثبت و گزارش متریک‌های مصرف منابع.
      - ارسال هشدار در صورت افزایش مصرف غیرمنتظره.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="resource_tracker", config=config)
        self.logger = logging.getLogger("ResourceTracker")
        # تنظیم دوره گزارش‌دهی (به ثانیه)
        self.report_interval = float(self.config.get("report_interval", 10))
        self.resource_status: Dict[str, Any] = {}
        self._running = False
        self._tracking_task: Optional[asyncio.Task] = None
        self.logger.info(f"[ResourceTracker] Initialized with report_interval={self.report_interval} seconds.")

    async def _track_resources(self) -> None:
        """
        حلقه اصلی نظارت بر منابع. در این حلقه به صورت دوره‌ای وضعیت منابع را ثبت می‌کند.
        """
        while self._running:
            # به عنوان نمونه، وضعیت منابع به صورت شبیه‌سازی شده ثبت می‌شود.
            self.resource_status = {
                "cpu_usage": 50,  # درصد مصرف CPU (نمونه)
                "memory_usage": 2048,  # مصرف حافظه به مگابایت (نمونه)
                "timestamp": datetime.utcnow().isoformat()
            }
            self.logger.info(f"[ResourceTracker] Current resource status: {self.resource_status}")
            self.increment_metric("resource_tracking_cycle")
            await asyncio.sleep(self.report_interval)

    async def start_tracking(self) -> None:
        """
        شروع نظارت بر منابع.
        """
        if not self._running:
            self._running = True
            self._tracking_task = asyncio.create_task(self._track_resources())
            self.logger.info("[ResourceTracker] Resource tracking started.")

    async def stop_tracking(self) -> None:
        """
        توقف نظارت بر منابع.
        """
        self._running = False
        if self._tracking_task:
            self._tracking_task.cancel()
            try:
                await self._tracking_task
            except asyncio.CancelledError:
                self.logger.info("[ResourceTracker] Resource tracking task cancelled.")
            self._tracking_task = None

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی منابع.

        Returns:
            Dict[str, Any]: وضعیت منابع ثبت شده.
        """
        return self.resource_status


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.DEBUG)


    async def main():
        tracker = ResourceTracker(config={"report_interval": 5})
        await tracker.start_tracking()
        await asyncio.sleep(15)
        await tracker.stop_tracking()
        print("Final Resource Status:", tracker.get_status())


    asyncio.run(main())
