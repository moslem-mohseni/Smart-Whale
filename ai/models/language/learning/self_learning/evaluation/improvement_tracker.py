"""
ImprovementTracker Module
---------------------------
این فایل مسئول پیگیری و ثبت پیشرفت مدل در طول زمان در فرآیند خودآموزی است.
کلاس ImprovementTracker اطلاعات دوره‌های آموزشی شامل متریک‌های عملکرد (مانند loss، accuracy، و ...) را به همراه زمان ثبت می‌کند
و امکان تولید گزارش‌های دوره‌ای و تاریخچه پیشرفت مدل را فراهم می‌کند.

این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import asyncio
import json
import logging
from abc import ABC
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..base.base_component import BaseComponent


class ImprovementTracker(BaseComponent, ABC):
    """
    ImprovementTracker مسئول ذخیره و پیگیری تاریخچه پیشرفت مدل در طول زمان است.

    ویژگی‌ها:
      - ثبت هر دوره آموزشی با تاریخچه متریک‌های کلیدی.
      - ذخیره تاریخچه پیشرفت به صورت درون حافظه و امکان ذخیره به فایل جهت بازیابی.
      - ارائه گزارش‌های خلاصه از روند بهبود مدل.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="improvement_tracker", config=config)
        self.logger = logging.getLogger("ImprovementTracker")
        # تاریخچه پیشرفت به صورت لیست از دیکشنری‌ها
        self.history: List[Dict[str, Any]] = []
        # مسیر پیش‌فرض فایل ذخیره‌سازی تاریخچه (در صورت نیاز)
        self.history_file = self.config.get("history_file", f"improvement_history_{self.component_id}.json")
        self.logger.info(f"[ImprovementTracker] Initialized. History file: {self.history_file}")

    def record_improvement(self, metrics: Dict[str, Any]) -> None:
        """
        ثبت اطلاعات پیشرفت یک دوره آموزشی.

        Args:
            metrics (Dict[str, Any]): دیکشنری شامل متریک‌های کلیدی مانند:
                - loss_before: مقدار loss قبل از آموزش
                - loss_after: مقدار loss بعد از آموزش
                - accuracy_before: دقت قبل از آموزش (اختیاری)
                - accuracy_after: دقت بعد از آموزش (اختیاری)
                - cycle_duration: مدت زمان اجرای دوره آموزشی به ثانیه
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }
        self.history.append(record)
        self.logger.info(f"[ImprovementTracker] Recorded improvement: {record}")
        self.increment_metric("improvement_recorded")

    def get_history(self) -> List[Dict[str, Any]]:
        """
        دریافت تاریخچه پیشرفت‌های ثبت‌شده.

        Returns:
            List[Dict[str, Any]]: لیست تاریخچه پیشرفت.
        """
        return self.history

    def generate_report(self) -> Dict[str, Any]:
        """
        تولید گزارش خلاصه از پیشرفت مدل بر اساس تاریخچه ثبت‌شده.
        گزارش شامل میانگین بهبود loss و تغییرات دقت می‌باشد.

        Returns:
            Dict[str, Any]: گزارش خلاصه پیشرفت.
        """
        total_cycles = len(self.history)
        if total_cycles == 0:
            self.logger.info("[ImprovementTracker] No improvement records to report.")
            return {"total_cycles": 0, "average_loss_improvement": 0.0, "accuracy_change": None,
                    "details": "No records available."}

        total_loss_improvement = 0.0
        accuracy_changes = []
        for record in self.history:
            m = record.get("metrics", {})
            loss_before = m.get("loss_before")
            loss_after = m.get("loss_after")
            if loss_before and loss_before > 0 and loss_after is not None:
                improvement = (loss_before - loss_after) / loss_before
                total_loss_improvement += improvement
            # در صورت وجود مقادیر دقت
            if "accuracy_before" in m and "accuracy_after" in m:
                acc_change = m["accuracy_after"] - m["accuracy_before"]
                accuracy_changes.append(acc_change)

        avg_loss_improvement = total_loss_improvement / total_cycles
        avg_accuracy_change = sum(accuracy_changes) / len(accuracy_changes) if accuracy_changes else None

        report = {
            "total_cycles": total_cycles,
            "average_loss_improvement": round(avg_loss_improvement, 6),
            "average_accuracy_change": round(avg_accuracy_change, 6) if avg_accuracy_change is not None else None,
            "details": f"Recorded {total_cycles} training cycles."
        }
        self.logger.info(f"[ImprovementTracker] Generated report: {report}")
        return report

    async def save_history(self, file_path: Optional[str] = None) -> bool:
        """
        ذخیره تاریخچه پیشرفت به فایل JSON.

        Args:
            file_path (Optional[str]): مسیر فایل؛ در صورت عدم تعیین از مسیر پیش‌فرض استفاده می‌شود.

        Returns:
            bool: True در صورت موفقیت‌آمیز بودن ذخیره‌سازی.
        """
        file_path = file_path or self.history_file
        try:
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[ImprovementTracker] History saved to {file_path}.")
            return True
        except Exception as e:
            self.logger.error(f"[ImprovementTracker] Error saving history: {str(e)}")
            self.record_error_metric()
            return False

    async def load_history(self, file_path: Optional[str] = None) -> bool:
        """
        بارگذاری تاریخچه پیشرفت از فایل JSON.

        Args:
            file_path (Optional[str]): مسیر فایل؛ در صورت عدم تعیین از مسیر پیش‌فرض استفاده می‌شود.

        Returns:
            bool: True در صورت موفقیت‌آمیز بودن بارگذاری.
        """
        file_path = file_path or self.history_file
        try:
            import os
            if not os.path.exists(file_path):
                self.logger.warning(f"[ImprovementTracker] History file not found at {file_path}.")
                return False
            with open(file_path, "r", encoding="utf-8") as f:
                self.history = json.load(f)
            self.logger.info(f"[ImprovementTracker] History loaded from {file_path}.")
            return True
        except Exception as e:
            self.logger.error(f"[ImprovementTracker] Error loading history: {str(e)}")
            self.record_error_metric()
            return False


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)


    async def main():
        tracker = ImprovementTracker()
        # شبیه‌سازی ثبت چند دوره آموزشی
        tracker.record_improvement({
            "loss_before": 0.5,
            "loss_after": 0.45,
            "cycle_duration": 100.0,
            "accuracy_before": 0.80,
            "accuracy_after": 0.82
        })
        tracker.record_improvement({
            "loss_before": 0.45,
            "loss_after": 0.42,
            "cycle_duration": 90.0,
            "accuracy_before": 0.82,
            "accuracy_after": 0.83
        })
        report = tracker.generate_report()
        print("Improvement Report:", report)
        # ذخیره و بارگذاری تاریخچه
        await tracker.save_history("improvement_history.json")
        tracker.history = []  # پاک کردن تاریخچه
        await tracker.load_history("improvement_history.json")
        print("Loaded History:", tracker.get_history())


    import asyncio

    asyncio.run(main())
