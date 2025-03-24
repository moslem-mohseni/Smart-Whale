from typing import Dict, Any


class AlertManager:
    """
    مدیریت هشدارهای سیستمی برای شناسایی مشکلات عملکردی مدل‌ها و منابع
    """

    def __init__(self):
        self.alerts: Dict[str, str] = {}  # ذخیره هشدارهای مربوط به مدل‌ها

    def generate_alert(self, model_name: str, metric: str, value: float, threshold: float) -> str:
        """
        تولید هشدار در صورت عبور مقدار متریک از حد آستانه تعیین‌شده
        :param model_name: نام مدل
        :param metric: نوع متریک (مثلاً latency, accuracy, throughput)
        :param value: مقدار متریک فعلی
        :param threshold: حد آستانه متریک
        :return: پیام هشدار به صورت رشته
        """
        if value > threshold:
            alert_msg = f"ALERT: {model_name} has {metric} exceeding threshold! Value: {value}, Threshold: {threshold}"
            self.alerts[model_name] = alert_msg
            return alert_msg
        return "No alert generated."

    def get_alerts(self) -> Dict[str, str]:
        """
        دریافت لیست هشدارهای ثبت‌شده
        :return: دیکشنری شامل هشدارهای ثبت‌شده برای مدل‌های مختلف
        """
        return self.alerts

    def clear_alerts(self) -> None:
        """
        حذف تمام هشدارهای ثبت‌شده در سیستم
        """
        self.alerts.clear()


# نمونه استفاده از AlertManager برای تست
if __name__ == "__main__":
    alert_manager = AlertManager()
    alert = alert_manager.generate_alert("model_a", "latency", 250, 200)
    print(alert)
    print(f"All Alerts: {alert_manager.get_alerts()}")
    alert_manager.clear_alerts()
    print(f"Alerts after clearing: {alert_manager.get_alerts()}")
