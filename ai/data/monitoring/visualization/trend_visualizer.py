# trend_visualizer.py
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List


class TrendVisualizer:
    """
    کلاس برای تحلیل روند متریک‌ها و نمایش تغییرات آن‌ها در طول زمان.
    """

    def __init__(self, output_dir: str = "trends/"):
        """
        مقداردهی اولیه کلاس.

        :param output_dir: دایرکتوری ذخیره نمودارهای روند.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_history: Dict[str, List[float]] = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": []
        }
        self.timestamps: List[str] = []

    def update_metrics(self, metrics_data: Dict[str, float]) -> None:
        """
        اضافه کردن مقادیر جدید به تاریخچه متریک‌ها.

        :param metrics_data: دیکشنری شامل متریک‌های جدید.
        """
        for key in self.metrics_history.keys():
            if key in metrics_data:
                self.metrics_history[key].append(metrics_data[key])

        self.timestamps.append(datetime.now().strftime("%H:%M:%S"))

        if len(self.timestamps) > 20:  # نگهداری آخرین 20 مقدار
            for key in self.metrics_history:
                self.metrics_history[key] = self.metrics_history[key][-20:]
            self.timestamps = self.timestamps[-20:]

    def plot_trends(self, filename: str = None) -> str:
        """
        رسم و ذخیره نمودارهای روند متریک‌ها.

        :param filename: نام فایل خروجی (در صورت عدم ورود، تاریخ فعلی انتخاب می‌شود).
        :return: مسیر ذخیره فایل تصویر.
        """
        filename = filename or f"trend_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        filepath = os.path.join(self.output_dir, filename)

        plt.figure(figsize=(10, 6))

        for metric, values in self.metrics_history.items():
            if values:
                plt.plot(self.timestamps, values, label=metric)

        plt.xlabel("زمان")
        plt.ylabel("مقدار متریک (%)")
        plt.title("📈 روند تغییرات متریک‌های سیستمی")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()

        plt.savefig(filepath)
        plt.close()
        print(f"✅ نمودار روند ذخیره شد: {filepath}")
        return filepath


if __name__ == "__main__":
    visualizer = TrendVisualizer()

    # شبیه‌سازی دریافت داده‌های متریک و رسم روند تغییرات
    for i in range(10):
        sample_metrics = {
            "cpu_usage": 50 + i * 2,
            "memory_usage": 60 + i * 1.5,
            "disk_io": 100 - i * 3,
            "network_io": 90 + i * 2
        }
        visualizer.update_metrics(sample_metrics)
        time.sleep(1)

    visualizer.plot_trends()
