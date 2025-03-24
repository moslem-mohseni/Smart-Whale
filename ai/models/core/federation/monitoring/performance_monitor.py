import time
from typing import Dict, Any, List
from infrastructure.timescaledb.timescale_manager import TimescaleDB


class PerformanceMonitor:
    """
    ماژول پایش عملکرد مدل‌های فدراسیونی، جمع‌آوری متریک‌های پردازشی و تحلیل روندها.
    """

    def __init__(self):
        """
        مقداردهی اولیه و تنظیم ارتباط با `TimescaleDB` برای ذخیره و تحلیل متریک‌های عملکردی.
        """
        self.timescale_db = TimescaleDB()
        self.performance_data: Dict[str, List[Dict[str, Any]]] = {}

    def record_performance(self, model_id: str, execution_time: float, success_rate: float,
                           resource_usage: Dict[str, float]):
        """
        ثبت داده‌های عملکردی یک مدل پس از اجرای وظیفه.
        :param model_id: شناسه مدل.
        :param execution_time: مدت زمان اجرای وظیفه (ثانیه).
        :param success_rate: نرخ موفقیت اجرای وظیفه (درصد).
        :param resource_usage: دیکشنری شامل مصرف CPU، حافظه و GPU.
        """
        timestamp = int(time.time())

        # ذخیره در پایگاه داده TimescaleDB
        self.timescale_db.store_timeseries(
            metric="model_performance",
            timestamp=timestamp,
            tags={"model_id": model_id},
            value={
                "execution_time": execution_time,
                "success_rate": success_rate,
                "cpu_usage": resource_usage["cpu"],
                "memory_usage": resource_usage["memory"],
                "gpu_usage": resource_usage["gpu"]
            }
        )

        # ذخیره محلی برای تحلیل روندها
        if model_id not in self.performance_data:
            self.performance_data[model_id] = []

        self.performance_data[model_id].append({
            "timestamp": timestamp,
            "execution_time": execution_time,
            "success_rate": success_rate,
            "resource_usage": resource_usage
        })

    def detect_anomalies(self, model_id: str) -> bool:
        """
        تشخیص ناهنجاری در عملکرد یک مدل بر اساس روندهای قبلی.
        :param model_id: شناسه مدل.
        :return: `True` اگر ناهنجاری تشخیص داده شود، در غیر اینصورت `False`.
        """
        if model_id not in self.performance_data or len(self.performance_data[model_id]) < 5:
            return False  # داده کافی برای تحلیل وجود ندارد

        recent_data = self.performance_data[model_id][-5:]
        avg_time = sum(d["execution_time"] for d in recent_data) / len(recent_data)
        avg_success = sum(d["success_rate"] for d in recent_data) / len(recent_data)

        if avg_time > 2.0 or avg_success < 80.0:
            return True  # اگر زمان اجرا بیش از حد طولانی یا نرخ موفقیت کم شده باشد، ناهنجاری محسوب می‌شود

        return False

    def get_performance_report(self, model_id: str) -> Dict[str, Any]:
        """
        دریافت گزارش عملکردی یک مدل فدراسیونی.
        :param model_id: شناسه مدل.
        :return: دیکشنری شامل اطلاعات عملکرد مدل.
        """
        if model_id not in self.performance_data:
            return {"error": "No performance data available for this model."}

        recent_data = self.performance_data[model_id][-10:]  # دریافت 10 رکورد آخر

        return {
            "model_id": model_id,
            "last_10_executions": recent_data
        }
