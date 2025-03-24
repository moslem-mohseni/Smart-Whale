import numpy as np
from scipy.stats import linregress
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor

class LoadPredictor:
    """
    ماژولی برای پیش‌بینی بار پردازشی آینده با استفاده از تحلیل روندهای گذشته.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
        self.history_window = 10  # تعداد نمونه‌های گذشته برای پیش‌بینی

    async def predict_load(self) -> dict:
        """
        پیش‌بینی میزان بار کاری در آینده.

        :return: دیکشنری شامل پیش‌بینی مصرف CPU، حافظه و I/O.
        """
        # دریافت متریک‌های تاریخی
        cpu_history = await self.metrics_collector.get_metric("cpu_usage", window=self.history_window)
        memory_history = await self.metrics_collector.get_metric("memory_usage", window=self.history_window)
        disk_io_history = await self.metrics_collector.get_metric("disk_io", window=self.history_window)

        # پیش‌بینی مقدار آینده
        cpu_prediction = self._predict_next_value(cpu_history)
        memory_prediction = self._predict_next_value(memory_history)
        disk_io_prediction = self._predict_next_value(disk_io_history)

        return {
            "cpu_prediction": cpu_prediction,
            "memory_prediction": memory_prediction,
            "disk_io_prediction": disk_io_prediction
        }

    def _predict_next_value(self, data_series: list) -> float:
        """
        پیش‌بینی مقدار آینده با استفاده از تحلیل روند خطی.

        :param data_series: لیستی از مقادیر تاریخی یک متریک خاص.
        :return: مقدار پیش‌بینی‌شده برای آینده.
        """
        if not data_series or len(data_series) < 2:
            return 0  # اگر داده کافی نداریم، مقدار 0 را برمی‌گردانیم

        # تولید شاخص‌های زمانی برای داده‌های تاریخی
        x = np.arange(len(data_series))
        y = np.array(data_series)

        # اجرای تحلیل رگرسیون خطی
        slope, intercept, _, _, _ = linregress(x, y)

        # پیش‌بینی مقدار آینده (زمان بعدی = x + 1)
        predicted_value = slope * (len(data_series)) + intercept
        return max(predicted_value, 0)  # مقدار منفی نباید برگردد
