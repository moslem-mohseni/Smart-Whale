import importlib
from typing import Dict, Any, Optional

class MLBasedGenerator:
    """
    این کلاس مسئول تولید پاسخ‌های مبتنی بر یادگیری ماشین است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای تولید پاسخ عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه تولیدکننده مبتنی بر یادگیری ماشین.
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
        تولید پاسخ با استفاده از مدل‌های یادگیری ماشین.
        :param text: متن ورودی
        :return: پاسخ تولیدشده
        """
        return self.language_processor.generate_ml_based_response(text)

    def process(self, text: str) -> Dict[str, Any]:
        """
        پردازش کامل تولید پاسخ مبتنی بر یادگیری ماشین.
        :param text: متن ورودی
        :return: دیکشنری شامل پاسخ تولیدشده
        """
        return {
            "language": self.language,
            "generated_response": self.generate_response(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    ml_generator = MLBasedGenerator(language="fa")

    text_sample_en = "Tell me about artificial intelligence."
    text_sample_fa = "درباره هوش مصنوعی به من بگو."
    text_sample_ru = "Расскажи мне об искусственном интеллекте."

    response_en = ml_generator.process(text_sample_en)
    response_fa = ml_generator.process(text_sample_fa)
    response_ru = ml_generator.process(text_sample_ru)

    print("🔹 English Response:")
    print(response_en)

    print("\n🔹 Persian Response:")
    print(response_fa)

    print("\n🔹 Russian Response:")
    print(response_ru)
