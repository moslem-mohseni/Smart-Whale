from typing import Dict


class PerformanceCollector:
    """
    جمع‌آوری متریک‌های عملکردی مدل‌ها شامل دقت، تأخیر و نرخ پردازش
    """

    def __init__(self):
        self.performance_data: Dict[str, Dict[str, float]] = {}  # داده‌های عملکرد مدل‌ها

    def collect_metrics(self, model_name: str, accuracy: float, latency: float, throughput: float) -> None:
        """
        جمع‌آوری متریک‌های عملکردی یک مدل خاص
        :param model_name: نام مدل
        :param accuracy: دقت مدل (بین 0 تا 1)
        :param latency: تأخیر پردازش (میلی‌ثانیه)
        :param throughput: نرخ پردازش (تعداد درخواست در ثانیه)
        """
        self.performance_data[model_name] = {
            "accuracy": accuracy,
            "latency": latency,
            "throughput": throughput
        }

    def get_metrics(self, model_name: str) -> Dict[str, float]:
        """
        دریافت متریک‌های عملکردی مدل مشخص‌شده
        :param model_name: نام مدل
        :return: دیکشنری شامل متریک‌های عملکرد مدل
        """
        return self.performance_data.get(model_name, {"accuracy": 0.0, "latency": 0.0, "throughput": 0.0})

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        دریافت متریک‌های عملکردی تمامی مدل‌ها
        :return: دیکشنری شامل متریک‌های عملکرد همه مدل‌ها
        """
        return self.performance_data


# نمونه استفاده از PerformanceCollector برای تست
if __name__ == "__main__":
    collector = PerformanceCollector()
    collector.collect_metrics("model_a", accuracy=0.92, latency=120.5, throughput=30.2)
    collector.collect_metrics("model_b", accuracy=0.88, latency=150.1, throughput=25.8)

    print(f"Performance Metrics for model_a: {collector.get_metrics('model_a')}")
    print(f"All Performance Metrics: {collector.get_all_metrics()}")
