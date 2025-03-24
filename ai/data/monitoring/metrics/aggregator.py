# aggregator.py
import time
import statistics
from typing import List, Dict

class MetricsAggregator:
    """
    کلاس برای تجمیع متریک‌های سیستم.
    این کلاس مقادیر متریک‌ها را در بازه‌های زمانی مشخص ذخیره و تجزیه‌وتحلیل می‌کند.
    """

    def __init__(self, retention_period: int = 60):
        """
        مقداردهی اولیه کلاس.

        :param retention_period: تعداد ثانیه‌هایی که داده‌های متریک ذخیره می‌شوند.
        """
        self.retention_period = retention_period
        self.metrics_history: Dict[str, List[float]] = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': []
        }

    def add_metrics(self, metrics_data: Dict[str, float]) -> None:
        """
        اضافه کردن متریک‌های جدید به تاریخچه.

        :param metrics_data: دیکشنری شامل متریک‌های جدید.
        """
        for key, value in metrics_data.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        self._prune_old_data()

    def get_aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        محاسبه متریک‌های تجمیعی و بازگرداندن آنها.

        :return: دیکشنری شامل میانگین، بیشینه و کمینه هر متریک.
        """
        aggregated_data = {}
        for key, values in self.metrics_history.items():
            if values:
                aggregated_data[key] = {
                    'average': round(statistics.mean(values), 2),
                    'max': max(values),
                    'min': min(values)
                }
            else:
                aggregated_data[key] = {'average': 0, 'max': 0, 'min': 0}
        return aggregated_data

    def _prune_old_data(self) -> None:
        """
        حذف داده‌های قدیمی از تاریخچه برای نگهداری بهینه.
        """
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > self.retention_period:
                self.metrics_history[key] = self.metrics_history[key][-self.retention_period:]

if __name__ == "__main__":
    aggregator = MetricsAggregator()

    # شبیه‌سازی دریافت متریک‌ها در بازه‌های زمانی
    for i in range(10):
        sample_metrics = {
            'cpu_usage': 30 + i * 0.5,
            'memory_usage': 60 + i * 0.3,
            'disk_io': 100 - i * 1.2,
            'network_io': 50 + i * 0.7
        }
        aggregator.add_metrics(sample_metrics)
        print(f"Aggregated Metrics: {aggregator.get_aggregated_metrics()}")
        time.sleep(5)
