import importlib
from typing import Dict, Any, Optional

class Rewriter:
    """
    این کلاس مسئول بازنویسی متن برای بهینه‌سازی و ساده‌سازی محتوا است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای بازنویسی عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None, level: str = "standard"):
        """
        مقداردهی اولیه بازنویس متن.
        :param language: زبان ورودی (در صورت `None`، زبان به‌طور خودکار شناسایی می‌شود)
        :param level: سطح بازنویسی (`simple`, `standard`, `creative`)
        """
        self.language = language
        self.level = level
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

    def rewrite_text(self, text: str) -> str:
        """
        بازنویسی متن ورودی با سطح مشخص‌شده.
        :param text: متن ورودی
        :return: متن بازنویسی‌شده
        """
        return self.language_processor.rewrite_text(text, self.level)

    def process(self, text: str) -> Dict[str, Any]:
        """
        پردازش کامل بازنویسی متن.
        :param text: متن ورودی
        :return: دیکشنری شامل متن بازنویسی‌شده
        """
        return {
            "language": self.language,
            "rewrite_level": self.level,
            "rewritten_text": self.rewrite_text(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    rewriter = Rewriter(language="fa", level="creative")

    text_sample_en = "Artificial intelligence is rapidly evolving, impacting various industries and changing the way we interact with technology."
    text_sample_fa = "هوش مصنوعی به سرعت در حال پیشرفت است و صنایع مختلف را تحت تأثیر قرار می‌دهد و نحوه تعامل ما با فناوری را تغییر می‌دهد."
    text_sample_ru = "Искусственный интеллект стремительно развивается, влияя на различные отрасли и изменяя наше взаимодействие с технологиями."

    rewritten_en = rewriter.process(text_sample_en)
    rewritten_fa = rewriter.process(text_sample_fa)
    rewritten_ru = rewriter.process(text_sample_ru)

    print("🔹 English Rewritten Text:")
    print(rewritten_en)

    print("\n🔹 Persian Rewritten Text:")
    print(rewritten_fa)

    print("\n🔹 Russian Rewritten Text:")
    print(rewritten_ru)
