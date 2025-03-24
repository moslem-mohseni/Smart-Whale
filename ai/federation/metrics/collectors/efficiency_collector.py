from core.monitoring.metrics.collector import Collector
from typing import Dict


class EfficiencyCollector(Collector):
    """
    جمع‌آوری متریک‌های بهره‌وری منابع از جمله مصرف CPU، حافظه و استفاده از GPU
    """

    def __init__(self):
        super().__init__()
        self.efficiency_data: Dict[str, Dict[str, float]] = {}  # داده‌های بهره‌وری مدل‌ها

    def collect_metrics(self, model_name: str, cpu_usage: float, memory_usage: float, gpu_usage: float) -> None:
        """
        جمع‌آوری متریک‌های بهره‌وری یک مدل خاص
        :param model_name: نام مدل
        :param cpu_usage: میزان مصرف CPU (درصد)
        :param memory_usage: میزان مصرف حافظه (مگابایت)
        :param gpu_usage: میزان استفاده از GPU (درصد)
        """
        self.efficiency_data[model_name] = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_usage": gpu_usage
        }
        super().collect_metrics(model_name, self.efficiency_data[model_name])

    def get_metrics(self, model_name: str) -> Dict[str, float]:
        """
        دریافت متریک‌های بهره‌وری یک مدل مشخص‌شده
        :param model_name: نام مدل
        :return: دیکشنری شامل متریک‌های بهره‌وری مدل یا مقدار پیش‌فرض
        """
        return self.efficiency_data.get(model_name, {"cpu_usage": 0.0, "memory_usage": 0.0, "gpu_usage": 0.0})

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        دریافت متریک‌های بهره‌وری تمامی مدل‌ها
        :return: دیکشنری شامل متریک‌های بهره‌وری همه مدل‌ها
        """
        return self.efficiency_data


# نمونه استفاده از EfficiencyCollector برای تست
if __name__ == "__main__":
    collector = EfficiencyCollector()
    collector.collect_metrics("model_a", cpu_usage=45.2, memory_usage=2048.5, gpu_usage=70.1)
    collector.collect_metrics("model_b", cpu_usage=60.3, memory_usage=4096.0, gpu_usage=85.7)

    print(f"Efficiency Metrics for model_a: {collector.get_metrics('model_a')}")
    print(f"All Efficiency Metrics: {collector.get_all_metrics()}")
