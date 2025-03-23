"""
MetricsCollector Module
-------------------------
این فایل مسئول جمع‌آوری، مدیریت و گزارش‌گیری متریک‌های عملکردی سیستم خودآموزی است.
این کلاس از کانترهای به‌روز (counter) به صورت ناهمزمان استفاده می‌کند و با به‌کارگیری یک وظیفه پس‌زمینه،
متریک‌ها را به صورت دوره‌ای ثبت و گزارش می‌دهد.
"""

import asyncio
import logging
from typing import Dict, Optional


class MetricsCollector:
    """
    کلاس MetricsCollector برای جمع‌آوری و گزارش متریک‌های عملکردی سیستم خودآموزی.

    ویژگی‌ها:
      - افزایش شمارنده‌ها (مانند تعداد درخواست‌ها و خطاها) به صورت ناهمزمان.
      - ثبت دوره‌ای متریک‌ها از طریق یک وظیفه پس‌زمینه جهت نظارت و گزارش.
      - ارائه متدهایی جهت دریافت وضعیت فعلی متریک‌ها.
    """

    def __init__(self, port: int = 9100, update_interval: int = 5):
        self.logger = logging.getLogger("MetricsCollector")
        self.port = port
        self.update_interval = update_interval
        self.metrics: Dict[str, float] = {
            "requests": 0,
            "errors": 0
        }
        self._lock = asyncio.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """
        شروع وظیفه پس‌زمینه جهت گزارش دوره‌ای متریک‌ها.
        """
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run())
            self.logger.info("[MetricsCollector] Background metric reporting started.")

    async def _run(self) -> None:
        """
        وظیفه پس‌زمینه جهت ثبت دوره‌ای و گزارش متریک‌ها.
        """
        while self._running:
            async with self._lock:
                # گزارش متریک‌ها در لاگ؛ می‌توان این قسمت را برای ارسال به سیستم‌های نظارتی مانند Prometheus توسعه داد.
                self.logger.debug(f"[MetricsCollector] Current Metrics: {self.metrics}")
            await asyncio.sleep(self.update_interval)

    async def stop(self) -> None:
        """
        توقف وظیفه پس‌زمینه و آزادسازی منابع.
        """
        self._running = False
        if self._task:
            await self._task
            self.logger.info("[MetricsCollector] Background metric reporting stopped.")

    async def increment_request_count(self) -> None:
        """
        افزایش شمارنده درخواست‌ها.
        """
        async with self._lock:
            self.metrics["requests"] += 1
            self.logger.debug(f"[MetricsCollector] Request count incremented to {self.metrics['requests']}.")

    async def increment_error_count(self) -> None:
        """
        افزایش شمارنده خطاها.
        """
        async with self._lock:
            self.metrics["errors"] += 1
            self.logger.debug(f"[MetricsCollector] Error count incremented to {self.metrics['errors']}.")

    async def record_metric(self, name: str, value: float) -> None:
        """
        ثبت یا افزایش یک متریک دلخواه.

        Args:
            name (str): نام متریک.
            value (float): مقدار افزایشی متریک.
        """
        async with self._lock:
            self.metrics[name] = self.metrics.get(name, 0) + value
            self.logger.debug(f"[MetricsCollector] Metric '{name}' updated to {self.metrics[name]}.")

    async def get_metrics(self) -> Dict[str, float]:
        """
        دریافت وضعیت فعلی متریک‌ها.

        Returns:
            Dict[str, float]: دیکشنری شامل مقادیر متریک‌ها.
        """
        async with self._lock:
            return dict(self.metrics)
