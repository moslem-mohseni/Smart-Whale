from typing import Dict, Any


class KnowledgeManager:
    """
    مدیریت دانش مدل‌ها از طریق ذخیره، بازیابی و به‌روزرسانی داده‌های مشترک
    """

    def __init__(self):
        self.knowledge_base: Dict[str, Any] = {}  # ذخیره دانش مدل‌ها

    def store_knowledge(self, model_name: str, knowledge: Any) -> None:
        """
        ذخیره دانش یک مدل در پایگاه دانش
        :param model_name: نام مدل هوش مصنوعی
        :param knowledge: داده‌های دانش برای ذخیره‌سازی
        """
        self.knowledge_base[model_name] = knowledge

    def retrieve_knowledge(self, model_name: str) -> Any:
        """
        بازیابی دانش یک مدل مشخص
        :param model_name: نام مدل موردنظر
        :return: دانش ذخیره‌شده یا None در صورت عدم وجود دانش
        """
        return self.knowledge_base.get(model_name, None)

    def update_knowledge(self, model_name: str, new_knowledge: Any) -> None:
        """
        به‌روزرسانی دانش یک مدل خاص
        :param model_name: نام مدل
        :param new_knowledge: داده‌های جدید برای به‌روزرسانی دانش
        """
        if model_name in self.knowledge_base:
            self.knowledge_base[model_name] = new_knowledge

    def delete_knowledge(self, model_name: str) -> None:
        """
        حذف دانش یک مدل از پایگاه دانش
        :param model_name: نام مدل
        """
        if model_name in self.knowledge_base:
            del self.knowledge_base[model_name]


# نمونه استفاده از KnowledgeManager برای تست
if __name__ == "__main__":
    manager = KnowledgeManager()
    manager.store_knowledge("model_a", {"accuracy": 0.95, "parameters": 12000})
    print(f"Retrieved Knowledge: {manager.retrieve_knowledge('model_a')}")
    manager.update_knowledge("model_a", {"accuracy": 0.96, "parameters": 12500})
    print(f"Updated Knowledge: {manager.retrieve_knowledge('model_a')}")
    manager.delete_knowledge("model_a")
    print(f"After Deletion: {manager.retrieve_knowledge('model_a')}")
