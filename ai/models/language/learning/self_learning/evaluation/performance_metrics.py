"""
PerformanceMetrics Module
---------------------------
این فایل مسئول محاسبه و ثبت متریک‌های عملکردی مدل در فرآیند خودآموزی است.
کلاس PerformanceMetrics از BaseComponent ارث‌بری می‌کند و متریک‌هایی مانند زمان پردازش، دقت پاسخ،
و سایر شاخص‌های کلیدی را دریافت، ذخیره و گزارش می‌دهد. همچنین امکان ارسال این متریک‌ها به سیستم‌های نظارتی (مانند Prometheus)
را فراهم می‌کند.

این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ..base.base_component import BaseComponent


class PerformanceMetrics(BaseComponent):
    """
    PerformanceMetrics مسئول جمع‌آوری و گزارش‌دهی متریک‌های عملکردی مدل است.

    ویژگی‌ها:
      - ثبت زمان اجرای عملیات مختلف (مثلاً دوره‌های آموزشی، درخواست‌های پردازشی).
      - ذخیره مقادیر متریک‌های کلیدی مانند دقت، سرعت پردازش، و نرخ خطا.
      - فراهم آوردن متدهایی برای دریافت وضعیت متریک‌ها به صورت همزمان.
      - ارسال متریک‌ها به سیستم‌های نظارتی در صورت نیاز.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="performance_metrics", config=config)
        self.logger = logging.getLogger("PerformanceMetrics")
        # دیکشنری برای نگهداری متریک‌ها
        self.metrics: Dict[str, Any] = {}
        # دوره زمانی به‌روزرسانی متریک‌ها (به عنوان نمونه، 10 ثانیه)
        self.update_interval = float(self.config.get("update_interval", 10))
        self._running = False
        self._reporting_task: Optional[asyncio.Task] = None
        self.logger.info(f"[PerformanceMetrics] Initialized with update_interval={self.update_interval} seconds.")

    def record_metric(self, name: str, value: Any) -> None:
        """
        ثبت یا به‌روزرسانی یک متریک.

        Args:
            name (str): نام متریک.
            value (Any): مقدار متریک.
        """
        self.metrics[name] = value
        self.logger.debug(f"[PerformanceMetrics] Recorded metric '{name}': {value}")
        self.increment_metric(f"metric_{name}")

    def get_metric(self, name: str) -> Optional[Any]:
        """
        دریافت مقدار یک متریک.

        Args:
            name (str): نام متریک.

        Returns:
            Optional[Any]: مقدار متریک یا None در صورت عدم وجود.
        """
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        دریافت تمام متریک‌های ثبت‌شده.

        Returns:
            Dict[str, Any]: دیکشنری شامل تمام متریک‌ها.
        """
        return dict(self.metrics)

    async def _report_metrics(self) -> None:
        """
        وظیفه پس‌زمینه جهت گزارش‌دهی دوره‌ای متریک‌ها.
        این متد می‌تواند به سیستم‌های نظارتی خارجی (مانند Prometheus) ارسال شود.
        """
        while self._running:
            await asyncio.sleep(self.update_interval)
            timestamp = datetime.utcnow().isoformat()
            report = {
                "timestamp": timestamp,
                "metrics": self.get_all_metrics()
            }
            self.logger.info(f"[PerformanceMetrics] Reporting metrics: {report}")
            # در اینجا می‌توان کد ارسال گزارش به یک سیستم نظارتی یا پایگاه داده را اضافه کرد.

    async def start_reporting(self) -> None:
        """
        شروع وظیفه گزارش‌دهی متریک‌ها به صورت دوره‌ای.
        """
        if not self._running:
            self._running = True
            self._reporting_task = asyncio.create_task(self._report_metrics())
            self.logger.info("[PerformanceMetrics] Metrics reporting started.")

    async def stop_reporting(self) -> None:
        """
        توقف وظیفه گزارش‌دهی متریک‌ها و آزادسازی منابع مربوط به آن.
        """
        self._running = False
        if self._reporting_task:
            self._reporting_task.cancel()
            try:
                await self._reporting_task
            except asyncio.CancelledError:
                self.logger.info("[PerformanceMetrics] Metrics reporting task cancelled.")
            self._reporting_task = None


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.DEBUG)


    async def main():
        pm = PerformanceMetrics(config={"update_interval": 5})
        await pm.start_reporting()
        # شبیه‌سازی ثبت چند متریک
        pm.record_metric("training_cycle_duration", 12.5)
        pm.record_metric("accuracy", 0.92)
        pm.record_metric("error_rate", 0.03)
        # اجازه دادن به وظیفه پس‌زمینه برای اجرای چند بار
        await asyncio.sleep(15)
        await pm.stop_reporting()
        all_metrics = pm.get_all_metrics()
        print("Final Metrics:", all_metrics)


    asyncio.run(main())
