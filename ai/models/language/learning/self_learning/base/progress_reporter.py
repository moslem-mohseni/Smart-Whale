"""
ProgressReporter Module
-------------------------
این فایل مسئول گزارش‌دهی و رصد پیشرفت کلی فرآیند خودآموزی است.
این کلاس به عنوان یک ابزار نظارتی عمل می‌کند که میزان پیشرفت مراحل مختلف چرخه یادگیری (مانند نیاز‌یابی، جمع‌آوری داده، پردازش، آموزش و ارزیابی)
را ذخیره، به‌روزرسانی و گزارش می‌دهد.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند، گزارش‌دهی دوره‌ای و ذخیره‌سازی وضعیت پیشرفت پیاده‌سازی شده است.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .base_component import BaseComponent


class ProgressReporter(BaseComponent):
    """
    کلاس ProgressReporter جهت مدیریت و گزارش‌دهی پیشرفت سیستم خودآموزی.

    ویژگی‌ها:
      - نگهداری یک دیکشنری از پیشرفت‌های هر مرحله.
      - به‌روزرسانی وضعیت پیشرفت با استفاده از متدهای بهینه و ناهمزمان.
      - گزارش‌دهی دوره‌ای پیشرفت‌ها به عنوان لاگ یا ذخیره‌سازی در فایل.
      - امکان ذخیره‌سازی وضعیت پیشرفت جهت بازیابی در صورت نیاز.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="progress_reporter", config=config)
        # دیکشنری پیشرفت، کلیدها نام مراحل مختلف چرخه یادگیری
        self.progress: Dict[str, float] = {
            "need_detection": 0.0,
            "acquisition": 0.0,
            "processing": 0.0,
            "training": 0.0,
            "evaluation": 0.0
        }
        self.report_interval = self.config.get("progress_report_interval", 10)  # به ثانیه
        self._reporting_task: Optional[asyncio.Task] = None
        self.logger.info("[ProgressReporter] Initialized with progress stages.")

    def update_progress(self, stage: str, value: float) -> None:
        """
        به‌روزرسانی درصد پیشرفت یک مرحله مشخص.

        Args:
            stage (str): نام مرحله (مثلاً "need_detection", "acquisition", "processing", "training", "evaluation").
            value (float): درصد پیشرفت (بین 0 تا 100).
        """
        if stage in self.progress:
            self.progress[stage] = max(0.0, min(100.0, value))
            self.logger.debug(f"[ProgressReporter] Updated '{stage}' progress to {self.progress[stage]}%.")
            self.increment_metric(f"progress_{stage}")
        else:
            self.logger.warning(f"[ProgressReporter] Unknown stage '{stage}' provided for progress update.")

    def get_progress(self) -> Dict[str, float]:
        """
        دریافت وضعیت فعلی پیشرفت مراحل.

        Returns:
            Dict[str, float]: دیکشنری شامل درصد پیشرفت هر مرحله.
        """
        return dict(self.progress)

    async def _periodic_report(self) -> None:
        """
        وظیفه پس‌زمینه جهت گزارش‌دهی دوره‌ای وضعیت پیشرفت.
        """
        while True:
            await asyncio.sleep(self.report_interval)
            progress_snapshot = self.get_progress()
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "progress": progress_snapshot
            }
            # در اینجا می‌توان گزارش را به سیستم‌های نظارتی یا فایل ذخیره کرد.
            self.logger.info(f"[ProgressReporter] Periodic Progress Report: {report}")
            # همچنین می‌توان از متریک‌های جمع‌آوری‌شده استفاده کرد.

    async def start_reporting(self) -> None:
        """
        شروع وظیفه گزارش‌دهی دوره‌ای.
        """
        if not self._reporting_task:
            self._reporting_task = asyncio.create_task(self._periodic_report())
            self.logger.info("[ProgressReporter] Periodic reporting started.")

    async def stop_reporting(self) -> None:
        """
        توقف وظیفه گزارش‌دهی دوره‌ای و آزادسازی منابع.
        """
        if self._reporting_task:
            self._reporting_task.cancel()
            try:
                await self._reporting_task
            except asyncio.CancelledError:
                self.logger.info("[ProgressReporter] Periodic reporting stopped.")
            self._reporting_task = None

    async def save_progress(self, file_path: Optional[str] = None) -> bool:
        """
        ذخیره وضعیت پیشرفت به صورت فایل JSON جهت بازیابی بعدی.

        Args:
            file_path (Optional[str]): مسیر فایل ذخیره‌سازی؛ در صورت عدم تعیین، مسیر پیش‌فرض استفاده می‌شود.

        Returns:
            bool: نتیجه موفقیت‌آمیز بودن ذخیره‌سازی.
        """
        file_path = file_path or f"progress_states/{self.component_id}_progress.json"
        try:
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            progress_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "progress": self.get_progress()
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[ProgressReporter] Progress saved to {file_path}.")
            return True
        except Exception as e:
            self.logger.error(f"[ProgressReporter] Failed to save progress: {str(e)}")
            self.record_error_metric()
            return False

    async def load_progress(self, file_path: Optional[str] = None) -> bool:
        """
        بارگذاری وضعیت پیشرفت از فایل ذخیره‌شده.

        Args:
            file_path (Optional[str]): مسیر فایل ذخیره‌شده؛ در صورت عدم تعیین، مسیر پیش‌فرض استفاده می‌شود.

        Returns:
            bool: نتیجه موفقیت‌آمیز بودن بارگذاری.
        """
        file_path = file_path or f"progress_states/{self.component_id}_progress.json"
        try:
            import os
            if not os.path.exists(file_path):
                self.logger.warning(f"[ProgressReporter] No progress file found at {file_path}.")
                return False
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_progress = data.get("progress", {})
            for stage, value in loaded_progress.items():
                if stage in self.progress:
                    self.progress[stage] = float(value)
            self.logger.info(f"[ProgressReporter] Progress loaded from {file_path}.")
            return True
        except Exception as e:
            self.logger.error(f"[ProgressReporter] Failed to load progress: {str(e)}")
            self.record_error_metric()
            return False
