from typing import List, Dict, Any


class AlertManager:
    """
    این کلاس مسئول مدیریت هشدارها در صورت کاهش کیفیت داده‌ها است.
    """

    def __init__(self, alert_threshold: float = 0.7):
        """
        مقداردهی اولیه با تنظیم آستانه‌ی هشدار کیفیت داده‌ها.
        """
        self.alert_threshold = alert_threshold  # حداقل میزان کیفیت قبل از ارسال هشدار
        self.alerts = []

    def check_for_alerts(self, quality_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        بررسی کیفیت داده‌ها و ایجاد هشدار در صورت کاهش کیفیت.
        """
        for data in quality_data:
            if data["quality_score"] < self.alert_threshold:
                alert = {
                    "data_id": data["data_id"],
                    "quality_score": data["quality_score"],
                    "message": "Quality below acceptable threshold!"
                }
                self.alerts.append(alert)
        return self.alerts

    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        دریافت هشدارهای ثبت‌شده.
        """
        return self.alerts

    def clear_alerts(self) -> None:
        """
        حذف تمام هشدارهای قبلی.
        """
        self.alerts.clear()
