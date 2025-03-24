from core.monitoring.metrics.collector import Collector
from typing import Dict


class QualityCollector(Collector):
    """
    جمع‌آوری متریک‌های کیفیت خروجی مدل‌ها شامل دقت، صحت و بازیابی
    """

    def __init__(self):
        super().__init__()
        self.quality_data: Dict[str, Dict[str, float]] = {}  # داده‌های کیفیت مدل‌ها

    def collect_metrics(self, model_name: str, accuracy: float, precision: float, recall: float) -> None:
        """
        جمع‌آوری متریک‌های کیفیت یک مدل خاص
        :param model_name: نام مدل
        :param accuracy: میزان دقت مدل (بین 0 تا 1)
        :param precision: میزان صحت مدل (بین 0 تا 1)
        :param recall: میزان بازیابی مدل (بین 0 تا 1)
        """
        self.quality_data[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }
        super().collect_metrics(model_name, self.quality_data[model_name])

    def get_metrics(self, model_name: str) -> Dict[str, float]:
        """
        دریافت متریک‌های کیفیت یک مدل مشخص‌شده
        :param model_name: نام مدل
        :return: دیکشنری شامل متریک‌های کیفیت مدل یا مقدار پیش‌فرض
        """
        return self.quality_data.get(model_name, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0})

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        دریافت متریک‌های کیفیت تمامی مدل‌ها
        :return: دیکشنری شامل متریک‌های کیفیت همه مدل‌ها
        """
        return self.quality_data


# نمونه استفاده از QualityCollector برای تست
if __name__ == "__main__":
    collector = QualityCollector()
    collector.collect_metrics("model_a", accuracy=0.92, precision=0.89, recall=0.87)
    collector.collect_metrics("model_b", accuracy=0.88, precision=0.85, recall=0.83)

    print(f"Quality Metrics for model_a: {collector.get_metrics('model_a')}")
    print(f"All Quality Metrics: {collector.get_all_metrics()}")
