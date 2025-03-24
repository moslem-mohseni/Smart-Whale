import importlib
from typing import Dict, Any, Optional

class LoadBalancer:
    """
    این کلاس مسئول مدیریت و متعادل‌سازی بار پردازشی سیستم پردازش زبان طبیعی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای توزیع عمومی بار پردازشی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None, balance_level: str = "standard"):
        """
        مقداردهی اولیه متعادل‌ساز بار پردازشی.
        :param language: زبان ورودی (در صورت `None`، زبان به‌طور خودکار شناسایی می‌شود)
        :param balance_level: سطح متعادل‌سازی بار (`low`, `standard`, `high`)
        """
        self.language = language
        self.balance_level = balance_level
        self.language_processor = self._load_processor()

    def _load_processor(self):
        """
        بررسی و بارگذاری ماژول پردازش زبان در صورت وجود.
        :return: ماژول پردازش زبان اختصاصی یا ماژول عمومی (`mBERT`) در صورت عدم وجود
        """
        try:
            module_path = f"ai.models.language.adaptors.{self.language}.language_processor"
            return importlib.import_module(module_path).LanguageProcessor()
        except ModuleNotFoundError:
            return importlib.import_module("ai.models.language.adaptors.multilingual.language_processor").LanguageProcessor()

    def distribute_load(self) -> Dict[str, Any]:
        """
        متعادل‌سازی بار پردازشی بر اساس حجم داده و میزان منابع موجود.
        :return: دیکشنری شامل وضعیت توزیع بار پردازشی
        """
        return self.language_processor.balance_load(self.balance_level)

    def process(self) -> Dict[str, Any]:
        """
        اجرای فرآیند متعادل‌سازی بار پردازشی در سیستم.
        :return: دیکشنری شامل اطلاعات توزیع بار
        """
        balance_result = self.distribute_load()

        return {
            "language": self.language,
            "balance_level": self.balance_level,
            "load_distribution": balance_result,
        }


# تست اولیه ماژول
if __name__ == "__main__":
    balancer = LoadBalancer(language="fa", balance_level="high")

    balance_fa = balancer.process()
    balance_en = LoadBalancer(language="en", balance_level="standard").process()
    balance_ru = LoadBalancer(language="ru", balance_level="low").process()

    print("🔹 Persian Load Balancing:")
    print(balance_fa)

    print("\n🔹 English Load Balancing:")
    print(balance_en)

    print("\n🔹 Russian Load Balancing:")
    print(balance_ru)
