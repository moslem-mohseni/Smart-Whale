from typing import Dict, List
from collections import Counter


class PatternDetector:
    """
    شناسایی الگوهای تکرارشونده در داده‌های متریک عملکردی مدل‌ها
    """

    def __init__(self):
        self.pattern_history: Dict[str, List[float]] = {}  # ذخیره تاریخچه متریک‌های مدل‌ها

    def record_metrics(self, model_name: str, metric_values: List[float]) -> None:
        """
        ثبت مقادیر متریک‌های یک مدل برای تحلیل الگوها
        :param model_name: نام مدل
        :param metric_values: لیست مقادیر متریک‌ها
        """
        if model_name not in self.pattern_history:
            self.pattern_history[model_name] = []
        self.pattern_history[model_name].extend(metric_values)

    def detect_pattern(self, model_name: str) -> str:
        """
        تحلیل داده‌های تاریخی یک مدل و شناسایی الگوهای تکراری
        :param model_name: نام مدل
        :return: توضیحی از الگوی شناسایی‌شده
        """
        if model_name not in self.pattern_history or len(self.pattern_history[model_name]) < 3:
            return "Not enough data to detect patterns."

        counter = Counter(self.pattern_history[model_name])
        most_common = counter.most_common(1)[0]

        if most_common[1] > len(self.pattern_history[model_name]) * 0.5:
            return f"Pattern detected: Frequent occurrence of {most_common[0]}"
        return "No significant pattern detected."


# نمونه استفاده از PatternDetector برای تست
if __name__ == "__main__":
    detector = PatternDetector()
    detector.record_metrics("model_a", [0.8, 0.85, 0.9, 0.85, 0.85, 0.88])
    print(f"Pattern Analysis for model_a: {detector.detect_pattern('model_a')}")
