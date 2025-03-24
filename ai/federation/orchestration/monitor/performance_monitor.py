from typing import Dict


class PerformanceMonitor:
    """
    پایش عملکرد مدل‌ها برای ارزیابی کارایی و شناسایی گلوگاه‌های پردازشی
    """

    def __init__(self):
        self.performance_data: Dict[str, Dict[str, float]] = {}  # نگهداری اطلاعات عملکرد هر مدل

    def update_performance(self, model_name: str, latency: float, throughput: float) -> None:
        """
        به‌روزرسانی داده‌های عملکرد یک مدل خاص
        :param model_name: نام مدل
        :param latency: میزان تأخیر پردازش (میلی‌ثانیه)
        :param throughput: نرخ پردازش (تعداد درخواست در ثانیه)
        """
        self.performance_data[model_name] = {"latency": latency, "throughput": throughput}

    def get_performance(self, model_name: str) -> Dict[str, float]:
        """
        دریافت داده‌های عملکرد مدل مشخص‌شده
        :param model_name: نام مدل
        :return: دیکشنری شامل میزان تأخیر و نرخ پردازش یا مقدار پیش‌فرض
        """
        return self.performance_data.get(model_name, {"latency": 0.0, "throughput": 0.0})

    def get_all_performance_data(self) -> Dict[str, Dict[str, float]]:
        """
        دریافت داده‌های عملکرد تمامی مدل‌ها
        :return: دیکشنری شامل اطلاعات عملکرد همه مدل‌ها
        """
        return self.performance_data


# نمونه استفاده از PerformanceMonitor برای تست
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.update_performance("model_a", latency=150.5, throughput=30.2)
    monitor.update_performance("model_b", latency=200.1, throughput=25.8)

    print(f"Performance Data for model_a: {monitor.get_performance('model_a')}")
    print(f"All Performance Data: {monitor.get_all_performance_data()}")
