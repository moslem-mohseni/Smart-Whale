import importlib
from typing import Dict, Any, Optional

class RuleBasedGenerator:
    """
    این کلاس مسئول تولید پاسخ‌های مبتنی بر قوانین است.
    اگر قوانین اختصاصی برای یک زبان موجود باشد، از `adaptors/` استفاده می‌کند.
    در غیر این صورت، از قوانین عمومی پیش‌فرض استفاده خواهد کرد.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه تولیدکننده مبتنی بر قوانین.
        :param language: زبان ورودی (در صورت `None`، زبان به‌طور خودکار شناسایی می‌شود)
        """
        self.language = language
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

    def generate_response(self, text: str) -> Dict[str, Any]:
        """
        تولید پاسخ مبتنی بر قوانین از پیش تعریف‌شده.
        :param text: متن ورودی
        :return: پاسخ تولیدشده
        """
        return self.language_processor.generate_rule_based_response(text)

    def process(self, text: str) -> Dict[str, Any]:
        """
        پردازش کامل تولید پاسخ مبتنی بر قوانین.
        :param text: متن ورودی
        :return: دیکشنری شامل پاسخ تولیدشده
        """
        return {
            "language": self.language,
            "generated_response": self.generate_response(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    rule_generator = RuleBasedGenerator(language="fa")

    text_sample_en = "Hello, how are you?"
    text_sample_fa = "سلام، حالت چطوره؟"
    text_sample_ru = "Привет, как дела?"

    response_en = rule_generator.process(text_sample_en)
    response_fa = rule_generator.process(text_sample_fa)
    response_ru = rule_generator.process(text_sample_ru)

    print("🔹 English Response:")
    print(response_en)

    print("\n🔹 Persian Response:")
    print(response_fa)

    print("\n🔹 Russian Response:")
    print(response_ru)
