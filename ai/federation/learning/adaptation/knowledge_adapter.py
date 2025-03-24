from typing import Dict, Any


class KnowledgeAdapter:
    """
    تنظیم و یکپارچه‌سازی دانش بین مدل‌های یادگیری فدراسیونی برای استفاده بهینه
    """

    def __init__(self):
        self.knowledge_registry: Dict[str, Any] = {}  # نگهداری داده‌های دانش برای مدل‌ها

    def register_knowledge(self, model_name: str, knowledge: Any) -> None:
        """
        ثبت دانش یک مدل برای به‌اشتراک‌گذاری بین مدل‌های دیگر
        :param model_name: نام مدل
        :param knowledge: داده‌های دانش مدل
        """
        self.knowledge_registry[model_name] = knowledge

    def get_knowledge(self, model_name: str) -> Any:
        """
        دریافت دانش یک مدل خاص
        :param model_name: نام مدل
        :return: دانش ذخیره‌شده یا مقدار None در صورت عدم وجود
        """
        return self.knowledge_registry.get(model_name, None)

    def transfer_knowledge(self, source_model: str, target_model: str) -> bool:
        """
        انتقال دانش از یک مدل به مدل دیگر
        :param source_model: نام مدل مبدا
        :param target_model: نام مدل مقصد
        :return: مقدار بولین که نشان‌دهنده موفقیت یا عدم موفقیت انتقال است
        """
        if source_model in self.knowledge_registry:
            self.knowledge_registry[target_model] = self.knowledge_registry[source_model]
            return True
        return False


# نمونه استفاده از KnowledgeAdapter برای تست
if __name__ == "__main__":
    adapter = KnowledgeAdapter()
    adapter.register_knowledge("model_a", {"features": [0.1, 0.2, 0.3], "accuracy": 0.95})
    print(f"Knowledge for model_a: {adapter.get_knowledge('model_a')}")
    success = adapter.transfer_knowledge("model_a", "model_b")
    print(f"Knowledge Transfer Successful: {success}")
    print(f"Knowledge for model_b: {adapter.get_knowledge('model_b')}")