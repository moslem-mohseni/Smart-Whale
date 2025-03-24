import importlib
from typing import Dict, Any, Optional

class QuantumAllocator:
    """
    این کلاس مسئول مدیریت و تخصیص بهینه منابع کوانتومی برای پردازش زبان طبیعی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای تخصیص عمومی منابع استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None, allocation_level: str = "standard"):
        """
        مقداردهی اولیه تخصیص‌دهنده کوانتومی.
        :param language: زبان ورودی (در صورت `None`، زبان به‌طور خودکار شناسایی می‌شود)
        :param allocation_level: سطح تخصیص منابع (`low`, `standard`, `high`)
        """
        self.language = language
        self.allocation_level = allocation_level
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

    def allocate_resources(self) -> Dict[str, Any]:
        """
        تخصیص بهینه منابع برای اجرای پردازش‌های کوانتومی.
        :return: دیکشنری شامل وضعیت تخصیص منابع
        """
        return self.language_processor.allocate_quantum_resources(self.allocation_level)

    def process(self) -> Dict[str, Any]:
        """
        اجرای فرآیند تخصیص منابع کوانتومی.
        :return: دیکشنری شامل اطلاعات تخصیص داده‌شده
        """
        allocation_result = self.allocate_resources()

        return {
            "language": self.language,
            "allocation_level": self.allocation_level,
            "resource_allocation": allocation_result,
        }


# تست اولیه ماژول
if __name__ == "__main__":
    allocator = QuantumAllocator(language="fa", allocation_level="high")

    allocation_fa = allocator.process()
    allocator_en = QuantumAllocator(language="en", allocation_level="standard").process()
    allocator_ru = QuantumAllocator(language="ru", allocation_level="low").process()

    print("🔹 Persian Quantum Resource Allocation:")
    print(allocation_fa)

    print("\n🔹 English Quantum Resource Allocation:")
    print(allocator_en)

    print("\n🔹 Russian Quantum Resource Allocation:")
    print(allocator_ru)
