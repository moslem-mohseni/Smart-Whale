from typing import Dict, List


class TrendAnalyzer:
    """
    تحلیل روند تغییرات متریک‌های عملکرد مدل‌ها در طول زمان
    """

    def __init__(self):
        self.trend_data: Dict[str, List[float]] = {}  # ذخیره تاریخچه متریک‌های مدل‌ها

    def record_metrics(self, model_name: str, metric_value: float) -> None:
        """
        ثبت مقدار متریک مدل در تاریخچه جهت تحلیل روند
        :param model_name: نام مدل
        :param metric_value: مقدار متریک ثبت‌شده
        """
        if model_name not in self.trend_data:
            self.trend_data[model_name] = []
        self.trend_data[model_name].append(metric_value)

    def analyze_trend(self, model_name: str) -> str:
        """
        تحلیل روند تغییرات متریک مدل مشخص‌شده
        :param model_name: نام مدل
        :return: توضیحی از روند شناسایی‌شده
        """
        if model_name not in self.trend_data or len(self.trend_data[model_name]) < 3:
            return "Not enough data to analyze trends."

        trend_values = self.trend_data[model_name]
        increasing = all(x < y for x, y in zip(trend_values, trend_values[1:]))
        decreasing = all(x > y for x, y in zip(trend_values, trend_values[1:]))

        if increasing:
            return "Upward trend detected. Model performance is improving."
        elif decreasing:
            return "Downward trend detected. Model performance is degrading."
        return "No significant trend detected."


# نمونه استفاده از TrendAnalyzer برای تست
if __name__ == "__main__":
    analyzer = TrendAnalyzer()
    analyzer.record_metrics("model_a", 0.75)
    analyzer.record_metrics("model_a", 0.78)
    analyzer.record_metrics("model_a", 0.82)
    print(f"Trend Analysis for model_a: {analyzer.analyze_trend('model_a')}")
