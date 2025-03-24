import importlib
from typing import Dict, Any, Optional

class AdaptivePipeline:
    """
    این کلاس مسئول مدیریت تطبیقی فرآیند پردازش متن است.
    بسته به نوع ورودی، مسیر پردازشی مناسب را انتخاب می‌کند و بهترین خروجی را تولید می‌کند.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه سیستم پردازش تطبیقی.
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

    def determine_processing_path(self, text: str) -> str:
        """
        تصمیم‌گیری در مورد مسیر پردازشی بسته به پیچیدگی متن.
        :param text: متن ورودی
        :return: مسیر پردازشی انتخاب‌شده (ساده، متوسط، پیچیده)
        """
        word_count = len(text.split())
        if word_count < 5:
            return "simple"
        elif 5 <= word_count < 20:
            return "intermediate"
        else:
            return "complex"

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        پردازش متن با انتخاب مسیر تطبیقی.
        :param text: متن ورودی
        :return: دیکشنری شامل تحلیل‌های انجام‌شده روی متن
        """
        processing_path = self.determine_processing_path(text)

        response = {"language": self.language, "processing_path": processing_path}

        if processing_path == "simple":
            response["syntax_analysis"] = self.language_processor.analyze_syntax(text)
            response["intent_detection"] = self.language_processor.detect_intent(text)

        elif processing_path == "intermediate":
            response["syntax_analysis"] = self.language_processor.analyze_syntax(text)
            response["semantic_analysis"] = self.language_processor.analyze_semantics(text)
            response["intent_detection"] = self.language_processor.detect_intent(text)
            response["named_entities"] = self.language_processor.recognize_entities(text)

        else:  # complex
            response["syntax_analysis"] = self.language_processor.analyze_syntax(text)
            response["semantic_analysis"] = self.language_processor.analyze_semantics(text)
            response["intent_detection"] = self.language_processor.detect_intent(text)
            response["named_entities"] = self.language_processor.recognize_entities(text)
            response["sentiment_analysis"] = self.language_processor.analyze_sentiment(text)
            response["context_analysis"] = self.language_processor.analyze_context()

        return response


# تست اولیه ماژول
if __name__ == "__main__":
    pipeline = AdaptivePipeline(language="fa")

    text_sample_simple = "سلام"
    text_sample_intermediate = "من دنبال یک مدل هوش مصنوعی برای تحلیل متن هستم."
    text_sample_complex = "نهنگ هوشمند پردازش زبان کوانتومی را درک می‌کند و می‌تواند درک عمیقی از ساختارهای نحوی، معنایی و زمینه‌ای متن ارائه دهد."

    result_simple = pipeline.process_text(text_sample_simple)
    result_intermediate = pipeline.process_text(text_sample_intermediate)
    result_complex = pipeline.process_text(text_sample_complex)

    print("\n🔹 Simple Processing:")
    print(result_simple)

    print("\n🔹 Intermediate Processing:")
    print(result_intermediate)

    print("\n🔹 Complex Processing:")
    print(result_complex)
