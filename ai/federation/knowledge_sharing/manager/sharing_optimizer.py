from typing import Dict, Any


class SharingOptimizer:
    """
    بهینه‌سازی فرآیند اشتراک دانش بین مدل‌ها برای کاهش سربار و افزایش کارایی
    """

    def __init__(self):
        self.optimized_data: Dict[str, Any] = {}  # نگهداری داده‌های بهینه‌شده برای اشتراک

    def optimize_sharing(self, model_name: str, knowledge: Any) -> Any:
        """
        بهینه‌سازی داده‌های دانش قبل از اشتراک‌گذاری برای کاهش سربار انتقال
        :param model_name: نام مدل ارائه‌دهنده دانش
        :param knowledge: داده‌های دانش مورد اشتراک‌گذاری
        :return: نسخه بهینه‌شده داده‌های دانش
        """
        optimized_knowledge = self._compress_knowledge(knowledge)
        self.optimized_data[model_name] = optimized_knowledge
        return optimized_knowledge

    def _compress_knowledge(self, knowledge: Any) -> Any:
        """
        فشرده‌سازی داده‌های دانش برای کاهش حجم انتقال
        :param knowledge: داده‌های ورودی
        :return: داده‌های فشرده‌شده
        """
        if isinstance(knowledge, dict):
            return {key: value for key, value in knowledge.items() if key in ["essential", "summary"]}
        return knowledge

    def get_optimized_knowledge(self, model_name: str) -> Any:
        """
        دریافت نسخه بهینه‌شده دانش یک مدل خاص
        :param model_name: نام مدل
        :return: دانش بهینه‌شده یا None در صورت عدم وجود داده
        """
        return self.optimized_data.get(model_name, None)


# نمونه استفاده از SharingOptimizer برای تست
if __name__ == "__main__":
    optimizer = SharingOptimizer()
    sample_knowledge = {"essential": "Core Info", "summary": "Brief Data", "extra": "Unnecessary Details"}

    optimized_knowledge = optimizer.optimize_sharing("model_a", sample_knowledge)
    print(f"Optimized Knowledge: {optimized_knowledge}")
    print(f"Retrieved Optimized Knowledge: {optimizer.get_optimized_knowledge('model_a')}")
