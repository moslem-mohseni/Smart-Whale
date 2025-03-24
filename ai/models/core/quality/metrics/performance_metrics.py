import time
from typing import Dict, Any, List
from infrastructure.timescaledb.timescale_manager import TimescaleDB

class PerformanceMetrics:
    """
    ماژول تحلیل متریک‌های عملکردی مدل‌های فدراسیونی شامل دقت، تأخیر، و مصرف منابع.
    """

    def __init__(self):
        """
        مقداردهی اولیه و تنظیم ارتباط با `TimescaleDB` برای ذخیره متریک‌های عملکردی.
        """
        self.timescale_db = TimescaleDB()
        self.metrics_data: Dict[str, List[Dict[str, Any]]] = {}

    def record_performance_metrics(self, model_id: str, accuracy: float, latency: float, resource_usage: Dict[str, float]):
        """
        ثبت متریک‌های عملکردی یک مدل فدراسیونی.
        :param model_id: شناسه مدل.
        :param accuracy: دقت مدل.
        :param latency: تأخیر پاسخ‌دهی مدل (میلی‌ثانیه).
        :param resource_usage: دیکشنری شامل مصرف CPU، حافظه، و GPU.
        """
        timestamp = int(time.time())

        # ذخیره در پایگاه داده TimescaleDB
        self.timescale_db.store_timeseries(
            metric="model_performance",
            timestamp=timestamp,
            tags={"model_id": model_id},
            value={
                "accuracy": accuracy,
                "latency": latency,
                "cpu_usage": resource_usage["cpu"],
                "memory_usage": resource_usage["memory"],
                "gpu_usage": resource_usage["gpu"]
            }
        )

        # ذخیره محلی برای تحلیل روندها
        if model_id not in self.metrics_data:
            self.metrics_data[model_id] = []

        self.metrics_data[model_id].append({
            "timestamp": timestamp,
            "accuracy": accuracy,
            "latency": latency,
            "resource_usage": resource_usage
        })

    def detect_performance_issues(self, model_id: str) -> bool:
        """
        تشخیص کاهش عملکرد در یک مدل بر اساس داده‌های اخیر.
        :param model_id: شناسه مدل.
        :return: `True` اگر کاهش عملکرد شناسایی شود، `False` در غیر اینصورت.
        """
        if model_id not in self.metrics_data or len(self.metrics_data[model_id]) < 5:
            return False  # داده کافی برای تحلیل وجود ندارد

        recent_data = self.metrics_data[model_id][-5:]
        avg_accuracy = sum(d["accuracy"] for d in recent_data) / len(recent_data)
        avg_latency = sum(d["latency"] for d in recent_data) / len(recent_data)

        if avg_accuracy < 80.0 or avg_latency > 500:
            return True  # اگر دقت کاهش یا تأخیر افزایش یابد، افت عملکرد محسوب می‌شود

        return False

    def get_performance_report(self, model_id: str) -> Dict[str, Any]:
        """
        دریافت گزارش عملکرد یک مدل فدراسیونی.
        :param model_id: شناسه مدل.
        :return: دیکشنری شامل اطلاعات عملکرد مدل.
        """
        if model_id not in self.metrics_data:
            return {"error": "No performance data available for this model."}

        recent_data = self.metrics_data[model_id][-10:]  # دریافت 10 رکورد آخر

        return {
            "model_id": model_id,
            "last_10_performance_records": recent_data
        }
