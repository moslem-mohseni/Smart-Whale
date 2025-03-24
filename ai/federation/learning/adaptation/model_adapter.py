from typing import Dict, List


class ModelAdapter:
    """
    سازگارسازی مدل‌های یادگیری فدراسیونی برای تطبیق دانش جدید با ساختار مدل‌های مختلف
    """

    def __init__(self):
        self.model_mappings: Dict[str, List[str]] = {}  # نگاشت مدل‌ها به نسخه‌های سازگار

    def register_model(self, model_name: str, compatible_models: List[str]) -> None:
        """
        ثبت یک مدل و مدل‌های سازگار با آن
        :param model_name: نام مدل اصلی
        :param compatible_models: لیست مدل‌های سازگار
        """
        self.model_mappings[model_name] = compatible_models

    def get_compatible_models(self, model_name: str) -> List[str]:
        """
        دریافت لیست مدل‌های سازگار با یک مدل خاص
        :param model_name: نام مدل
        :return: لیست مدل‌های سازگار
        """
        return self.model_mappings.get(model_name, [])

    def is_compatible(self, source_model: str, target_model: str) -> bool:
        """
        بررسی سازگاری بین دو مدل مختلف
        :param source_model: نام مدل مبدا
        :param target_model: نام مدل مقصد
        :return: مقدار بولین که نشان‌دهنده سازگار بودن یا نبودن مدل‌هاست
        """
        return target_model in self.model_mappings.get(source_model, [])


# نمونه استفاده از ModelAdapter برای تست
if __name__ == "__main__":
    adapter = ModelAdapter()
    adapter.register_model("model_a", ["model_b", "model_c"])
    print(f"Compatible Models for model_a: {adapter.get_compatible_models('model_a')}")
    print(f"Is model_a compatible with model_b? {adapter.is_compatible('model_a', 'model_b')}")
    print(f"Is model_a compatible with model_d? {adapter.is_compatible('model_a', 'model_d')}")

