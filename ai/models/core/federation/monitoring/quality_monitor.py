import time
from typing import Dict, Any, List
from infrastructure.timescaledb.timescale_manager import TimescaleDB

class QualityMonitor:
    """
    ماژول پایش کیفیت خروجی مدل‌های فدراسیونی و تحلیل روند کیفیت مدل‌ها.
    """

    def __init__(self):
        """
        مقداردهی اولیه و تنظیم ارتباط با `TimescaleDB` برای ذخیره و تحلیل متریک‌های کیفیت.
        """
        self.timescale_db = TimescaleDB()
        self.quality_data: Dict[str, List[Dict[str, Any]]] = {}

    def record_quality_metrics(self, model_id: str, accuracy: float, precision: float, recall: float, coherence: float):
        """
        ثبت متریک‌های کیفیت خروجی مدل‌های فدراسیونی.
        :param model_id: شناسه مدل.
        :param accuracy: دقت مدل.
        :param precision: صحت مدل.
        :param recall: بازیابی اطلاعات مدل.
        :param coherence: انسجام پاسخ‌های مدل.
        """
        timestamp = int(time.time())

        # ذخیره در پایگاه داده TimescaleDB
        self.timescale_db.store_timeseries(
            metric="model_quality",
            timestamp=timestamp,
            tags={"model_id": model_id},
            value={
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "coherence": coherence
            }
        )

        # ذخیره محلی برای تحلیل روندها
        if model_id not in self.quality_data:
            self.quality_data[model_id] = []

        self.quality_data[model_id].append({
            "timestamp": timestamp,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "coherence": coherence
        })

    def detect_quality_degradation(self, model_id: str) -> bool:
        """
        تشخیص افت کیفیت در عملکرد مدل.
        :param model_id: شناسه مدل.
        :return: `True` اگر افت کیفیت شناسایی شود، `False` در غیر اینصورت.
        """
        if model_id not in self.quality_data or len(self.quality_data[model_id]) < 5:
            return False  # داده کافی برای تحلیل وجود ندارد

        recent_data = self.quality_data[model_id][-5:]
        avg_accuracy = sum(d["accuracy"] for d in recent_data) / len(recent_data)
        avg_coherence = sum(d["coherence"] for d in recent_data) / len(recent_data)

        if avg_accuracy < 75.0 or avg_coherence < 70.0:
            return True  # اگر دقت یا انسجام کاهش یابد، افت کیفیت محسوب می‌شود

        return False

    def get_quality_report(self, model_id: str) -> Dict[str, Any]:
        """
        دریافت گزارش کیفیت عملکردی یک مدل فدراسیونی.
        :param model_id: شناسه مدل.
        :return: دیکشنری شامل اطلاعات کیفیت مدل.
        """
        if model_id not in self.quality_data:
            return {"error": "No quality data available for this model."}

        recent_data = self.quality_data[model_id][-10:]  # دریافت 10 رکورد آخر

        return {
            "model_id": model_id,
            "last_10_quality_reports": recent_data
        }
