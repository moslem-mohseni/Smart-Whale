from typing import Dict, Any


class OrchestrationOptimizer:
    """
    بهینه‌سازی هماهنگی بین مدل‌ها برای افزایش بهره‌وری پردازش
    """

    def __init__(self):
        self.optimization_data: Dict[str, Any] = {}  # داده‌های مربوط به بهینه‌سازی هماهنگی

    def analyze_workflow(self, model_name: str, execution_time: float, resource_usage: float) -> None:
        """
        تحلیل روند اجرای مدل و ذخیره داده‌های مربوط به بهینه‌سازی
        :param model_name: نام مدل
        :param execution_time: زمان اجرای پردازش (ثانیه)
        :param resource_usage: میزان استفاده از منابع (واحد پردازنده)
        """
        self.optimization_data[model_name] = {
            "execution_time": execution_time,
            "resource_usage": resource_usage
        }

    def get_optimization_data(self, model_name: str) -> Dict[str, float]:
        """
        دریافت اطلاعات بهینه‌سازی یک مدل خاص
        :param model_name: نام مدل
        :return: دیکشنری شامل زمان اجرا و میزان مصرف منابع
        """
        return self.optimization_data.get(model_name, {"execution_time": 0.0, "resource_usage": 0.0})

    def suggest_improvement(self, model_name: str) -> str:
        """
        ارائه پیشنهاد برای بهبود عملکرد مدل بر اساس داده‌های بهینه‌سازی
        :param model_name: نام مدل
        :return: پیشنهادات بهینه‌سازی به صورت رشته
        """
        data = self.optimization_data.get(model_name)
        if not data:
            return "No optimization data available."

        if data["execution_time"] > 5.0:
            return "Consider parallel processing or model pruning to reduce execution time."
        elif data["resource_usage"] > 80.0:
            return "Optimize resource allocation or use more efficient data structures."
        else:
            return "Model is performing optimally."


# نمونه استفاده از OrchestrationOptimizer برای تست
if __name__ == "__main__":
    optimizer = OrchestrationOptimizer()
    optimizer.analyze_workflow("model_a", execution_time=6.5, resource_usage=85.0)
    optimizer.analyze_workflow("model_b", execution_time=3.2, resource_usage=50.0)

    print(f"Optimization Data for model_a: {optimizer.get_optimization_data('model_a')}")
    print(f"Suggested Improvement for model_a: {optimizer.suggest_improvement('model_a')}")
    print(f"Suggested Improvement for model_b: {optimizer.suggest_improvement('model_b')}")
