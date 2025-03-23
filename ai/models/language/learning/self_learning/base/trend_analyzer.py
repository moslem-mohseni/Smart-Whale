"""
TrendAnalyzer Module
----------------------
این فایل مسئول تحلیل روند تغییرات متریک‌های عملکردی سیستم خودآموزی است.
این ماژول با استفاده از تاریخچه (history) متریک‌ها و الگوریتم خطی‌سازی (linear regression)
شیب روند هر متریک را محاسبه می‌کند تا تشخیص دهد که آیا متریک در حال افزایش، کاهش یا ثابت بودن است.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Dict, Deque, Tuple, Any, Optional

import numpy as np


class TrendAnalyzer:
    """
    کلاس TrendAnalyzer مسئول تحلیل روند متریک‌های سیستم خودآموزی است.

    ویژگی‌ها:
      - نگهداری تاریخچه (history) برای هر متریک با استفاده از پنجره لغزان (sliding window).
      - تحلیل روند با استفاده از الگوریتم خطی‌سازی (linear regression) برای محاسبه شیب.
      - تعیین روند به عنوان "increasing"، "decreasing" یا "stable" بر اساس مقدار شیب و آستانه تعیین شده.
    """

    def __init__(self, window_size: int = 30, slope_threshold: float = 0.01) -> None:
        """
        مقداردهی اولیه TrendAnalyzer.

        Args:
            window_size (int): تعداد نمونه‌های تاریخی برای هر متریک.
            slope_threshold (float): آستانه برای تشخیص تغییر معنادار در شیب.
        """
        self.logger = logging.getLogger("TrendAnalyzer")
        self.window_size = window_size
        self.slope_threshold = slope_threshold
        # نگهداری تاریخچه برای هر متریک: نام متریک -> deque از (زمان، مقدار)
        self.metric_history: Dict[str, Deque[Tuple[datetime, float]]] = {}
        self.logger.info(
            f"[TrendAnalyzer] Initialized with window_size={window_size} and slope_threshold={slope_threshold}")

    def update_metric(self, metric_name: str, value: float) -> None:
        """
        به‌روزرسانی تاریخچه یک متریک با مقدار جدید.

        Args:
            metric_name (str): نام متریک.
            value (float): مقدار جدید متریک.
        """
        now = datetime.utcnow()
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = deque(maxlen=self.window_size)
        self.metric_history[metric_name].append((now, value))
        self.logger.debug(f"[TrendAnalyzer] Updated '{metric_name}' with value {value} at {now.isoformat()}.")

    def analyze_trend(self, metric_name: str) -> Dict[str, Optional[Any]]:
        """
        تحلیل روند یک متریک بر اساس تاریخچه جمع‌آوری شده.

        Args:
            metric_name (str): نام متریک.

        Returns:
            Dict[str, Optional[Any]]: شامل:
                - trend: "increasing", "decreasing" یا "stable"
                - slope: مقدار شیب محاسبه‌شده (اگر قابل محاسبه نباشد None)
                - count: تعداد نمونه‌های موجود در تاریخچه
        """
        history = self.metric_history.get(metric_name)
        if not history or len(history) < 2:
            self.logger.debug(f"[TrendAnalyzer] Not enough data to analyze trend for '{metric_name}'.")
            return {"trend": None, "slope": None, "count": len(history) if history else 0}

        # استخراج زمان‌ها به صورت ثانیه از اولین نمونه و مقادیر
        times = [(t - history[0][0]).total_seconds() for t, _ in history]
        values = [v for _, v in history]

        try:
            p = np.polyfit(times, values, 1)
            slope, intercept = p[0], p[1]
            if slope > self.slope_threshold:
                trend = "increasing"
            elif slope < -self.slope_threshold:
                trend = "decreasing"
            else:
                trend = "stable"
            self.logger.debug(f"[TrendAnalyzer] Analyzed '{metric_name}': slope={slope}, trend={trend}")
            return {"trend": trend, "slope": slope, "count": len(history)}
        except Exception as e:
            self.logger.error(f"[TrendAnalyzer] Error analyzing trend for '{metric_name}': {str(e)}")
            return {"trend": None, "slope": None, "count": len(history)}

    async def analyze_all_trends(self) -> Dict[str, Dict[str, Optional[Any]]]:
        """
        تحلیل روند تمامی متریک‌های ثبت‌شده.

        Returns:
            Dict[str, Dict[str, Optional[Any]]]: دیکشنری که کلید آن نام متریک و مقدار آن نتایج تحلیل روند است.
        """
        trends = {}
        for metric in self.metric_history.keys():
            trends[metric] = self.analyze_trend(metric)
        self.logger.info(f"[TrendAnalyzer] Completed analysis for all metrics: {trends}")
        return trends


# نمونه تستی برای TrendAnalyzer
if __name__ == "__main__":
    async def main():
        analyzer = TrendAnalyzer(window_size=10, slope_threshold=0.05)
        # به‌روزرسانی چند نمونه برای متریک "accuracy"
        for i in range(15):
            # شبیه‌سازی افزایش تدریجی دقت
            analyzer.update_metric("accuracy", 0.80 + 0.005 * i)
            await asyncio.sleep(0.1)
        result = analyzer.analyze_trend("accuracy")
        print("Trend analysis for 'accuracy':", result)
        all_trends = await analyzer.analyze_all_trends()
        print("All trend results:", all_trends)


    asyncio.run(main())
