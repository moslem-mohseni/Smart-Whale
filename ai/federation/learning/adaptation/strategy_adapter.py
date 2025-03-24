from typing import Dict


class StrategyAdapter:
    """
    بهینه‌سازی استراتژی‌های یادگیری بین مدل‌ها برای تطبیق بهتر با داده‌های جدید
    """

    def __init__(self):
        self.strategy_registry: Dict[str, str] = {}  # نگهداری استراتژی‌های یادگیری مدل‌ها

    def register_strategy(self, model_name: str, strategy: str) -> None:
        """
        ثبت استراتژی یادگیری برای یک مدل خاص
        :param model_name: نام مدل
        :param strategy: نام استراتژی یادگیری
        """
        self.strategy_registry[model_name] = strategy

    def get_strategy(self, model_name: str) -> str:
        """
        دریافت استراتژی یادگیری مدل مشخص‌شده
        :param model_name: نام مدل
        :return: استراتژی ذخیره‌شده یا مقدار پیش‌فرض "unknown"
        """
        return self.strategy_registry.get(model_name, "unknown")

    def update_strategy(self, model_name: str, new_strategy: str) -> None:
        """
        به‌روزرسانی استراتژی یادگیری یک مدل
        :param model_name: نام مدل
        :param new_strategy: نام استراتژی جدید
        """
        if model_name in self.strategy_registry:
            self.strategy_registry[model_name] = new_strategy


# نمونه استفاده از StrategyAdapter برای تست
if __name__ == "__main__":
    adapter = StrategyAdapter()
    adapter.register_strategy("model_a", "supervised")
    print(f"Strategy for model_a: {adapter.get_strategy('model_a')}")
    adapter.update_strategy("model_a", "unsupervised")
    print(f"Updated Strategy for model_a: {adapter.get_strategy('model_a')}")
