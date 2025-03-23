"""
PatternDetector Module
------------------------
این فایل مسئول تشخیص الگوهای خاص و تغییرات ناگهانی در متریک‌های سیستم خودآموزی است.
این ماژول با استفاده از الگوریتم‌های ساده آماری (مانند محاسبه میانگین متحرک و انحراف معیار) به تشخیص انحرافات و الگوهای غیرعادی می‌پردازد.
این نسخه نهایی و عملیاتی است و باید در سیستم‌های تولیدی با بهترین کارایی و هوشمندی به کار رود.
"""

import asyncio
import logging
import statistics
from collections import deque
from typing import Dict, Deque, Optional, Any


class PatternDetector:
    """
    کلاس PatternDetector مسئول نظارت بر دنباله‌های زمانی متریک‌ها و تشخیص الگوهای غیرعادی در آن‌هاست.

    ویژگی‌ها:
      - نگهداری یک پنجره لغزان (sliding window) برای هر متریک جهت محاسبه میانگین و انحراف معیار.
      - تشخیص انحرافات بر اساس فاصله استاندارد (z-score) از میانگین.
      - ارائه متدهایی برای به‌روزرسانی متریک‌ها و دریافت گزارش‌های تشخیص.
      - قابلیت تنظیم اندازه پنجره و آستانه تشخیص (threshold) به صورت پویا.
    """

    def __init__(self, window_size: int = 20, z_threshold: float = 2.5) -> None:
        """
        مقداردهی اولیه PatternDetector.

        Args:
            window_size (int): تعداد نمونه‌های ذخیره شده در پنجره لغزان برای هر متریک.
            z_threshold (float): آستانه ز-اسکور برای تشخیص انحراف غیرمعمول.
        """
        self.logger = logging.getLogger("PatternDetector")
        self.window_size = window_size
        self.z_threshold = z_threshold
        # نگهداری پنجره‌های لغزان برای هر متریک؛ هر متریک یک deque از اعداد (float)
        self.metric_windows: Dict[str, Deque[float]] = {}
        self.logger.info(f"[PatternDetector] Initialized with window_size={window_size} and z_threshold={z_threshold}")

    def update_metric(self, metric_name: str, value: float) -> None:
        """
        به‌روزرسانی پنجره لغزان یک متریک با مقدار جدید.

        Args:
            metric_name (str): نام متریک.
            value (float): مقدار جدید متریک.
        """
        if metric_name not in self.metric_windows:
            self.metric_windows[metric_name] = deque(maxlen=self.window_size)
        self.metric_windows[metric_name].append(value)
        self.logger.debug(
            f"[PatternDetector] Updated '{metric_name}' with value {value}. Current window: {list(self.metric_windows[metric_name])}")

    def detect_anomaly(self, metric_name: str) -> Dict[str, Any]:
        """
        تشخیص انحراف غیرمعمول (آنومالی) برای یک متریک بر اساس پنجره لغزان.

        Args:
            metric_name (str): نام متریک.

        Returns:
            Dict[str, Any]: دیکشنری شامل:
                - anomaly (bool): آیا انحراف غیرمعمول تشخیص داده شده است.
                - z_score (Optional[float]): مقدار z-اسکور آخرین مقدار، در صورت محاسبه.
                - mean (Optional[float]): میانگین پنجره.
                - std_dev (Optional[float]): انحراف معیار پنجره.
                - details (str): توضیح مختصر از نتیجه تشخیص.
        """
        window = self.metric_windows.get(metric_name)
        if not window or len(window) < 2:
            self.logger.debug(f"[PatternDetector] Not enough data for metric '{metric_name}' to detect anomaly.")
            return {
                "anomaly": False,
                "z_score": None,
                "mean": None,
                "std_dev": None,
                "details": "Not enough data"
            }
        try:
            mean_val = statistics.mean(window)
            std_dev = statistics.stdev(window)
            latest = window[-1]
            # در صورتی که انحراف معیار صفر باشد، نمی‌توان z-score محاسبه کرد.
            if std_dev == 0:
                z_score = 0.0
            else:
                z_score = abs((latest - mean_val) / std_dev)
            anomaly = z_score >= self.z_threshold
            details = "Anomaly detected" if anomaly else "Normal pattern"
            self.logger.debug(
                f"[PatternDetector] Metric '{metric_name}': latest={latest}, mean={mean_val}, std_dev={std_dev}, z_score={z_score}")
            return {
                "anomaly": anomaly,
                "z_score": z_score,
                "mean": mean_val,
                "std_dev": std_dev,
                "details": details
            }
        except Exception as e:
            self.logger.error(f"[PatternDetector] Error detecting anomaly for '{metric_name}': {str(e)}")
            return {
                "anomaly": False,
                "z_score": None,
                "mean": None,
                "std_dev": None,
                "details": f"Error: {str(e)}"
            }

    async def analyze_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        تحلیل تمامی متریک‌های ذخیره‌شده و تشخیص الگوهای غیرمعمول برای هر یک.

        Returns:
            Dict[str, Dict[str, Any]]: دیکشنری‌ای که کلید آن نام متریک و مقدار آن نتایج تشخیص انحراف است.
        """
        results = {}
        for metric_name in self.metric_windows.keys():
            results[metric_name] = self.detect_anomaly(metric_name)
        self.logger.info(f"[PatternDetector] Completed analysis for metrics: {results}")
        return results
