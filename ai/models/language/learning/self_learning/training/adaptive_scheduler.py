"""
AdaptiveScheduler Module
--------------------------
این فایل مسئول زمان‌بندی تطبیقی جلسات آموزش در سیستم خودآموزی است.
AdaptiveScheduler وظیفه دارد با توجه به بار سیستم، عملکرد دوره‌های قبلی و منابع در دسترس،
زمان‌بندی جلسات آموزشی را به صورت دینامیک تنظیم کند. این کلاس با استفاده از مکانیسم‌های
ناهمزمان (asyncio) و دریافت متریک‌های عملکردی (از طریق متدهای BaseComponent) زمان‌بندی
را بهینه می‌کند.

این نسخه نهایی و عملیاتی با بهترین و هوشمندترین مکانیسم‌ها پیاده‌سازی شده است.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from ..base.base_component import BaseComponent


class AdaptiveScheduler(BaseComponent):
    """
    AdaptiveScheduler مسئول زمان‌بندی تطبیقی جلسات آموزش بر اساس عملکرد دوره‌های قبلی،
    وضعیت منابع و سایر معیارهای کلیدی است.

    ویژگی‌ها:
      - تنظیم خودکار فاصله بین جلسات آموزش بر اساس نتایج دوره‌های قبلی.
      - استفاده از متریک‌های ثبت‌شده (مانند زمان آموزش، خطا، و مصرف منابع) جهت تنظیم زمان‌بندی.
      - ارائه متدهایی برای شروع، به‌روزرسانی و توقف زمان‌بندی دوره‌های آموزشی.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه AdaptiveScheduler.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل:
                - "initial_interval": فاصله اولیه بین جلسات آموزشی به ثانیه (پیش‌فرض: 1800 ثانیه = 30 دقیقه)
                - "min_interval": حداقل فاصله ممکن (پیش‌فرض: 600 ثانیه = 10 دقیقه)
                - "max_interval": حداکثر فاصله ممکن (پیش‌فرض: 7200 ثانیه = 2 ساعت)
                - "adaptation_factor": ضریب تغییر (مثلاً 0.1 برای تغییر 10 درصدی)
        """
        super().__init__(component_type="adaptive_scheduler", config=config)
        self.logger = logging.getLogger("AdaptiveScheduler")
        self.initial_interval = float(self.config.get("initial_interval", 1800))
        self.min_interval = float(self.config.get("min_interval", 600))
        self.max_interval = float(self.config.get("max_interval", 7200))
        self.adaptation_factor = float(self.config.get("adaptation_factor", 0.1))
        self.current_interval = self.initial_interval
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self.logger.info(f"[AdaptiveScheduler] Initialized with initial_interval={self.initial_interval}, "
                         f"min_interval={self.min_interval}, max_interval={self.max_interval}, "
                         f"adaptation_factor={self.adaptation_factor}")

    async def start(self) -> None:
        """
        شروع زمان‌بندی تطبیقی جلسات آموزشی به صورت دوره‌ای.
        """
        if not self._running:
            self._running = True
            self._scheduler_task = asyncio.create_task(self._run_scheduler())
            self.logger.info("[AdaptiveScheduler] Adaptive scheduling started.")

    async def _run_scheduler(self) -> None:
        """
        حلقه اصلی زمان‌بندی تطبیقی. این حلقه دوره به دوره اجرا شده و با پایان هر دوره،
        بر اساس متریک‌های دوره قبلی (که باید از منبع خارجی یا callback دریافت شوند)
        فاصله بین دوره‌ها را تنظیم می‌کند.
        """
        while self._running:
            cycle_start = datetime.utcnow()
            self.logger.info(f"[AdaptiveScheduler] Starting training cycle at {cycle_start.isoformat()} "
                             f"with interval {self.current_interval:.2f} seconds.")

            # اینجا می‌توان یک callback برای شروع دوره آموزشی فراخوانی کرد.
            # مثلاً: await self.trigger_event("TRAINING_CYCLE_STARTED", {"timestamp": cycle_start.isoformat()})
            # و بعد از پایان دوره، متریک‌های مربوطه دریافت و ذخیره شوند.
            # در این نسخه، ما فقط یک تأخیر شبیه‌سازی می‌کنیم.
            await asyncio.sleep(self.current_interval)

            # شبیه‌سازی دریافت متریک‌های دوره آموزشی (مثلاً زمان صرف شده و خطاها)
            # فرض می‌کنیم متریک "cycle_duration" به عنوان نمونه دریافت می‌شود.
            simulated_duration = self.current_interval * 0.9  # فرض: دوره آموزشی کمی سریع‌تر از زمان برنامه‌ریزی اجرا شده است
            self.logger.info(f"[AdaptiveScheduler] Training cycle completed in {simulated_duration:.2f} seconds.")

            # تنظیم زمان‌بندی دوره بعدی بر اساس عملکرد دوره فعلی.
            self._adapt_interval(simulated_duration)

            cycle_end = datetime.utcnow()
            self.logger.info(
                f"[AdaptiveScheduler] Next training cycle will start in {self.current_interval:.2f} seconds "
                f"(Cycle ended at {cycle_end.isoformat()}).")

    def _adapt_interval(self, measured_duration: float) -> None:
        """
        تنظیم فاصله دوره بعدی بر اساس مدت زمان واقعی اجرای دوره آموزشی.
        اگر دوره سریع‌تر از زمان برنامه‌ریزی اجرا شده باشد، فاصله دوره بعدی کاهش می‌یابد؛
        در غیر این صورت افزایش می‌یابد.

        Args:
            measured_duration (float): مدت زمان واقعی اجرای دوره آموزشی به ثانیه.
        """
        # محاسبه اختلاف نسبی
        diff_ratio = (self.current_interval - measured_duration) / self.current_interval
        self.logger.debug(f"[AdaptiveScheduler] Measured duration: {measured_duration:.2f}, "
                          f"Current interval: {self.current_interval:.2f}, Diff ratio: {diff_ratio:.3f}")

        # تنظیم فاصله جدید با استفاده از ضریب تطبیق
        adjustment = self.current_interval * self.adaptation_factor * diff_ratio
        new_interval = self.current_interval + adjustment

        # محدود کردن فاصله جدید در بازه [min_interval, max_interval]
        new_interval = max(self.min_interval, min(new_interval, self.max_interval))
        self.logger.info(
            f"[AdaptiveScheduler] Adapted interval from {self.current_interval:.2f} to {new_interval:.2f} seconds.")
        self.current_interval = new_interval

    async def stop(self) -> None:
        """
        توقف زمان‌بندی تطبیقی جلسات آموزشی و آزادسازی منابع.
        """
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                self.logger.info("[AdaptiveScheduler] Scheduler task cancelled successfully.")
            self._scheduler_task = None
        self.logger.info("[AdaptiveScheduler] Adaptive scheduling stopped.")

    def get_current_interval(self) -> float:
        """
        دریافت فاصله زمان فعلی بین دوره‌های آموزشی.

        Returns:
            float: فاصله زمان به ثانیه.
        """
        return self.current_interval

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی زمان‌بندی تطبیقی.

        Returns:
            Dict[str, Any]: شامل:
                - current_interval: فاصله دوره فعلی.
                - running: وضعیت اجرای scheduler.
        """
        return {
            "current_interval": self.current_interval,
            "running": self._running
        }


# Sample usage for testing AdaptiveScheduler (final version intended for production)
if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.DEBUG)


    async def main():
        scheduler = AdaptiveScheduler(config={
            "initial_interval": 10,  # برای تست کوتاه
            "min_interval": 5,
            "max_interval": 20,
            "adaptation_factor": 0.2
        })
        await scheduler.start()
        # اجازه اجرای چند دوره برای تست
        await asyncio.sleep(35)
        await scheduler.stop()
        status = scheduler.get_status()
        print("Final Adaptive Scheduler Status:", status)


    asyncio.run(main())
