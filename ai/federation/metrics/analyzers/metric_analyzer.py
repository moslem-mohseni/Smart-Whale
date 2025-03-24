from typing import Dict, Any


class MetricAnalyzer:
    """
    تحلیل متریک‌های کل سیستم و تشخیص ناهنجاری‌ها در عملکرد مدل‌ها
    """

    def __init__(self):
        self.analyzed_data: Dict[str, Dict[str, Any]] = {}  # ذخیره تحلیل‌های متریک‌های سیستم

    def analyze_metrics(self, model_name: str, metrics: Dict[str, float]) -> str:
        """
        تحلیل متریک‌های یک مدل و تشخیص مشکلات احتمالی
        :param model_name: نام مدل
        :param metrics: دیکشنری شامل متریک‌های عملکرد مدل
        :return: نتیجه تحلیل به‌صورت رشته
        """
        self.analyzed_data[model_name] = metrics

        if metrics["latency"] > 200:
            return "High latency detected. Consider optimization."
        elif metrics["accuracy"] < 0.7:
            return "Low accuracy detected. Consider retraining the model."
        elif metrics["throughput"] < 10:
            return "Low throughput detected. Consider scaling the system."
        else:
            return "Model performance is within acceptable limits."

    def get_analysis(self, model_name: str) -> Dict[str, Any]:
        """
        دریافت تحلیل‌های متریک مدل مشخص‌شده
        :param model_name: نام مدل
        :return: دیکشنری شامل تحلیل متریک‌های مدل یا مقدار پیش‌فرض
        """
        return self.analyzed_data.get(model_name, {"status": "No analysis available"})


# نمونه استفاده از MetricAnalyzer برای تست
if __name__ == "__main__":
    analyzer = MetricAnalyzer()
    metrics = {"accuracy": 0.65, "latency": 250, "throughput": 8}
    analysis_result = analyzer.analyze_metrics("model_a", metrics)
    print(f"Analysis Result for model_a: {analysis_result}")
    print(f"Stored Analysis: {analyzer.get_analysis('model_a')}")
