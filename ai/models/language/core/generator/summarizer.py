import importlib
from typing import Dict, Any, Optional

class Summarizer:
    """
    این کلاس مسئول خلاصه‌سازی متن ورودی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای خلاصه‌سازی عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None, method: str = "extractive"):
        """
        مقداردهی اولیه خلاصه‌ساز.
        :param language: زبان ورودی (در صورت `None`، زبان به‌طور خودکار شناسایی می‌شود)
        :param method: روش خلاصه‌سازی (extractive: انتخاب جملات کلیدی، abstractive: تولید خلاصه جدید)
        """
        self.language = language
        self.method = method
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

    def extractive_summarization(self, text: str) -> str:
        """
        خلاصه‌سازی استخراجی: انتخاب جملات کلیدی از متن.
        :param text: متن ورودی
        :return: خلاصه استخراجی
        """
        return self.language_processor.extractive_summarization(text)

    def abstractive_summarization(self, text: str) -> str:
        """
        خلاصه‌سازی انتزاعی: تولید خلاصه‌ای جدید بر اساس درک محتوا.
        :param text: متن ورودی
        :return: خلاصه انتزاعی
        """
        return self.language_processor.abstractive_summarization(text)

    def summarize(self, text: str) -> Dict[str, Any]:
        """
        پردازش خلاصه‌سازی متن بر اساس روش انتخاب‌شده.
        :param text: متن ورودی
        :return: دیکشنری شامل خلاصه تولیدشده
        """
        summary = self.extractive_summarization(text) if self.method == "extractive" else self.abstractive_summarization(text)

        return {
            "language": self.language,
            "method": self.method,
            "summary": summary,
        }


# تست اولیه ماژول
if __name__ == "__main__":
    summarizer = Summarizer(language="fa", method="abstractive")

    text_sample_en = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to human intelligence. 
    AI applications include expert systems, natural language processing, speech recognition, and machine vision.
    AI research has led to the development of autonomous systems capable of performing tasks that normally 
    require human intelligence, such as decision-making, pattern recognition, and problem-solving.
    """

    text_sample_fa = """
    هوش مصنوعی (AI) شاخه‌ای از علوم کامپیوتر است که به توسعه سیستم‌هایی اختصاص دارد که قادر به انجام کارهایی هستند که به طور معمول به هوش انسانی نیاز دارند.
    این شامل پردازش زبان طبیعی، تشخیص گفتار، بینایی ماشین و سیستم‌های خبره می‌شود.
    تحقیقات هوش مصنوعی به توسعه سیستم‌های خودمختار منجر شده است که قادر به تصمیم‌گیری، تشخیص الگوها و حل مسائل هستند.
    """

    text_sample_ru = """
    Искусственный интеллект (ИИ) — это интеллект, демонстрируемый машинами, в отличие от человеческого интеллекта.
    Приложения ИИ включают экспертные системы, обработку естественного языка, распознавание речи и машинное зрение.
    Исследования в области ИИ привели к разработке автономных систем, способных выполнять задачи, требующие человеческого интеллекта.
    """

    summary_en = summarizer.summarize(text_sample_en)
    summary_fa = summarizer.summarize(text_sample_fa)
    summary_ru = summarizer.summarize(text_sample_ru)

    print("🔹 English Summary:")
    print(summary_en)

    print("\n🔹 Persian Summary:")
    print(summary_fa)

    print("\n🔹 Russian Summary:")
    print(summary_ru)
