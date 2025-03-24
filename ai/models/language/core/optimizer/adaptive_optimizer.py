import importlib
from typing import Dict, Any, Optional

class AdaptiveOptimizer:
    """
    این کلاس مسئول بهینه‌سازی تطبیقی پردازش زبان طبیعی است.
    بسته به نوع ورودی، مسیر پردازشی مناسب را انتخاب کرده و بهترین منابع را تخصیص می‌دهد.
    """

    def __init__(self, language: Optional[str] = None, optimization_level: str = "standard"):
        """
        مقداردهی اولیه سیستم بهینه‌سازی تطبیقی.
        :param language: زبان ورودی (در صورت `None`، زبان به‌طور خودکار شناسایی می‌شود)
        :param optimization_level: سطح بهینه‌سازی (`low`, `standard`, `high`)
        """
        self.language = language
        self.optimization_level = optimization_level
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

    def optimize_processing_path(self, text: str) -> str:
        """
        تعیین مسیر پردازشی بهینه بسته به پیچیدگی متن.
        :param text: متن ورودی
        :return: مسیر پردازشی انتخاب‌شده (`light`, `balanced`, `advanced`)
        """
        word_count = len(text.split())
        if word_count < 5:
            return "light"
        elif 5 <= word_count < 20:
            return "balanced"
        else:
            return "advanced"

    def allocate_optimized_resources(self) -> Dict[str, Any]:
        """
        تخصیص بهینه منابع برای اجرای پردازش‌های زبان طبیعی.
        :return: دیکشنری شامل وضعیت تخصیص منابع
        """
        return self.language_processor.allocate_optimized_resources(self.optimization_level)

    def process(self, text: str) -> Dict[str, Any]:
        """
        اجرای فرآیند بهینه‌سازی پردازش زبان طبیعی.
        :param text: متن ورودی
        :return: دیکشنری شامل اطلاعات پردازش بهینه‌شده
        """
        processing_path = self.optimize_processing_path(text)
        allocation_result = self.allocate_optimized_resources()

        return {
            "language": self.language,
            "processing_path": processing_path,
            "optimization_level": self.optimization_level,
            "resource_allocation": allocation_result,
        }


# تست اولیه ماژول
if __name__ == "__main__":
    optimizer = AdaptiveOptimizer(language="fa", optimization_level="high")

    text_sample_simple = "سلام"
    text_sample_intermediate = "من به دنبال یادگیری هوش مصنوعی هستم."
    text_sample_complex = "پردازش زبان طبیعی یکی از چالش‌برانگیزترین حوزه‌های هوش مصنوعی است که شامل یادگیری عمیق، پردازش کوانتومی و بهینه‌سازی منابع پردازشی می‌شود."

    result_simple = optimizer.process(text_sample_simple)
    result_intermediate = optimizer.process(text_sample_intermediate)
    result_complex = optimizer.process(text_sample_complex)

    print("\n🔹 Simple Processing Optimization:")
    print(result_simple)

    print("\n🔹 Intermediate Processing Optimization:")
    print(result_intermediate)

    print("\n🔹 Complex Processing Optimization:")
    print(result_complex)
