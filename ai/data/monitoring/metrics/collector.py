# collector.py
import time
import prometheus_client
from typing import Dict

class MetricsCollector:
    """
    کلاس برای جمع‌آوری متریک‌های سیستم.
    این کلاس متریک‌ها را از بخش‌های مختلف سیستم جمع‌آوری و در Prometheus ثبت می‌کند.
    """

    def __init__(self):
        # ثبت متریک‌های پایه
        self.cpu_usage = prometheus_client.Gauge('cpu_usage', 'CPU Usage of the System')
        self.memory_usage = prometheus_client.Gauge('memory_usage', 'Memory Usage of the System')
        self.disk_io = prometheus_client.Gauge('disk_io', 'Disk I/O Operations')
        self.network_io = prometheus_client.Gauge('network_io', 'Network I/O Operations')

    def collect(self) -> Dict[str, float]:
        """
        جمع‌آوری متریک‌ها و بازگرداندن به صورت دیکشنری.
        """
        metrics_data = {
            'cpu_usage': self._get_cpu_usage(),
            'memory_usage': self._get_memory_usage(),
            'disk_io': self._get_disk_io(),
            'network_io': self._get_network_io()
        }
        self._update_prometheus_metrics(metrics_data)
        return metrics_data

    def _get_cpu_usage(self) -> float:
        """
        محاسبه و بازگرداندن میزان استفاده از CPU.
        """
        # این متد به‌صورت نمونه پیاده‌سازی شده و باید با کد واقعی سیستم جایگزین شود
        return 30.0  # مقدار نمونه

    def _get_memory_usage(self) -> float:
        """
        محاسبه و بازگرداندن میزان استفاده از حافظه.
        """
        # این متد به‌صورت نمونه پیاده‌سازی شده و باید با کد واقعی سیستم جایگزین شود
        return 60.0  # مقدار نمونه

    def _get_disk_io(self) -> float:
        """
        محاسبه و بازگرداندن میزان عملیات ورودی/خروجی دیسک.
        """
        # این متد به‌صورت نمونه پیاده‌سازی شده و باید با کد واقعی سیستم جایگزین شود
        return 100.0  # مقدار نمونه

    def _get_network_io(self) -> float:
        """
        محاسبه و بازگرداندن میزان عملیات ورودی/خروجی شبکه.
        """
        # این متد به‌صورت نمونه پیاده‌سازی شده و باید با کد واقعی سیستم جایگزین شود
        return 50.0  # مقدار نمونه

    def _update_prometheus_metrics(self, metrics_data: Dict[str, float]) -> None:
        """
        به‌روزرسانی مقادیر متریک‌ها در Prometheus.
        """
        self.cpu_usage.set(metrics_data['cpu_usage'])
        self.memory_usage.set(metrics_data['memory_usage'])
        self.disk_io.set(metrics_data['disk_io'])
        self.network_io.set(metrics_data['network_io'])

if __name__ == "__main__":
    collector = MetricsCollector()
    while True:
        data = collector.collect()
        print(f"Collected Metrics: {data}")
        time.sleep(10)  # جمع‌آوری متریک‌ها هر 10 ثانیه
