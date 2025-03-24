from typing import Dict


class QualityMonitor:
    """
    پایش کیفیت خروجی مدل‌های هوش مصنوعی برای اطمینان از عملکرد مطلوب
    """

    def __init__(self):
        self.quality_metrics: Dict[str, Dict[str, float]] = {}  # نگهداری معیارهای کیفیت برای هر مدل

    def update_quality_metrics(self, model_name: str, accuracy: float, precision: float, recall: float) -> None:
        """
        به‌روزرسانی معیارهای کیفیت برای یک مدل خاص
        :param model_name: نام مدل
        :param accuracy: دقت مدل (بین 0 تا 1)
        :param precision: صحت مدل (بین 0 تا 1)
        :param recall: بازیابی مدل (بین 0 تا 1)
        """
        self.quality_metrics[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }

    def get_quality_metrics(self, model_name: str) -> Dict[str, float]:
        """
        دریافت معیارهای کیفیت مدل مشخص‌شده
        :param model_name: نام مدل
        :return: دیکشنری شامل دقت، صحت و بازیابی مدل یا مقدار پیش‌فرض
        """
        return self.quality_metrics.get(model_name, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0})

    def get_all_quality_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        دریافت معیارهای کیفیت تمامی مدل‌های ثبت‌شده
        :return: دیکشنری شامل معیارهای کیفیت همه مدل‌ها
        """
        return self.quality_metrics


# نمونه استفاده از QualityMonitor برای تست
if __name__ == "__main__":
    monitor = QualityMonitor()
    monitor.update_quality_metrics("model_a", accuracy=0.92, precision=0.89, recall=0.87)
    monitor.update_quality_metrics("model_b", accuracy=0.88, precision=0.85, recall=0.83)

    print(f"Quality Metrics for model_a: {monitor.get_quality_metrics('model_a')}")
    print(f"All Quality Metrics: {monitor.get_all_quality_metrics()}")
