from typing import Dict, Any


class MetricOptimizer:
    """
    بهینه‌سازی متریک‌های عملکردی سیستم برای افزایش کارایی مدل‌ها
    """

    def __init__(self):
        self.optimization_data: Dict[str, Dict[str, Any]] = {}  # نگهداری داده‌های بهینه‌سازی

    def optimize_metrics(self, model_name: str, metrics: Dict[str, float]) -> str:
        """
        بهینه‌سازی متریک‌های یک مدل بر اساس معیارهای کارایی
        :param model_name: نام مدل
        :param metrics: دیکشنری شامل متریک‌های مدل
        :return: نتیجه بهینه‌سازی به‌صورت رشته
        """
        self.optimization_data[model_name] = metrics

        if metrics["latency"] > 200:
            return "Optimization needed: Reduce model complexity or enable caching."
        elif metrics["accuracy"] < 0.7:
            return "Optimization needed: Improve training data or adjust hyperparameters."
        elif metrics["throughput"] < 10:
            return "Optimization needed: Scale up resources or optimize inference engine."
        else:
            return "Model metrics are optimal."

    def get_optimization_data(self, model_name: str) -> Dict[str, Any]:
        """
        دریافت داده‌های بهینه‌سازی یک مدل خاص
        :param model_name: نام مدل
        :return: دیکشنری شامل اطلاعات بهینه‌سازی مدل
        """
        return self.optimization_data.get(model_name, {"status": "No optimization data available"})


# نمونه استفاده از MetricOptimizer برای تست
if __name__ == "__main__":
    optimizer = MetricOptimizer()
    metrics = {"accuracy": 0.65, "latency": 250, "throughput": 8}
    optimization_result = optimizer.optimize_metrics("model_a", metrics)
    print(f"Optimization Result for model_a: {optimization_result}")
    print(f"Stored Optimization Data: {optimizer.get_optimization_data('model_a')}")
