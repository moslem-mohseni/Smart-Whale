from typing import Dict


class HealthMonitor:
    """
    پایش سلامت مدل‌ها و منابع پردازشی برای تشخیص مشکلات و بهینه‌سازی عملکرد
    """

    def __init__(self):
        self.model_health: Dict[str, str] = {}  # وضعیت سلامت مدل‌ها (healthy, degraded, failed)

    def update_health_status(self, model_name: str, status: str) -> None:
        """
        به‌روزرسانی وضعیت سلامت یک مدل خاص
        :param model_name: نام مدل
        :param status: وضعیت سلامت (healthy, degraded, failed)
        """
        if status not in ["healthy", "degraded", "failed"]:
            raise ValueError("Invalid health status")

        self.model_health[model_name] = status

    def get_health_status(self, model_name: str) -> str:
        """
        دریافت وضعیت سلامت مدل مشخص‌شده
        :param model_name: نام مدل
        :return: وضعیت سلامت مدل یا مقدار پیش‌فرض "unknown"
        """
        return self.model_health.get(model_name, "unknown")

    def get_all_health_statuses(self) -> Dict[str, str]:
        """
        دریافت وضعیت سلامت همه مدل‌های ثبت‌شده
        :return: دیکشنری شامل وضعیت سلامت تمامی مدل‌ها
        """
        return self.model_health


# نمونه استفاده از HealthMonitor برای تست
if __name__ == "__main__":
    monitor = HealthMonitor()
    monitor.update_health_status("model_a", "healthy")
    monitor.update_health_status("model_b", "degraded")

    print(f"Health Status for model_a: {monitor.get_health_status('model_a')}")
    print(f"All Health Statuses: {monitor.get_all_health_statuses()}")
