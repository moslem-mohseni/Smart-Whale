import importlib
from typing import Dict, Any, Optional, List

class FeatureExtractor:
    """
    این کلاس مسئول استخراج ویژگی‌های زبانی از متن ورودی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای تحلیل عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه استخراج ویژگی‌های زبانی.
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

    def extract_syntax_features(self, text: str) -> List[Dict[str, Any]]:
        """
        استخراج ویژگی‌های نحوی مانند برچسب‌های دستوری (POS) و وابستگی‌ها.
        :param text: متن ورودی
        :return: لیستی از ویژگی‌های نحوی متن
        """
        return self.language_processor.analyze_syntax(text)

    def extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """
        استخراج ویژگی‌های معنایی متن مانند مفهوم کلی و روابط معنایی.
        :param text: متن ورودی
        :return: دیکشنری از ویژگی‌های معنایی
        """
        return self.language_processor.analyze_semantics(text)

    def extract_text_complexity(self, text: str) -> Dict[str, float]:
        """
        تحلیل پیچیدگی متن از جمله میانگین طول جملات، میانگین طول کلمات و تنوع واژگانی.
        :param text: متن ورودی
        :return: دیکشنری شامل معیارهای پیچیدگی متن
        """
        sentences = text.split(".")
        words = text.split()

        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        avg_word_length = sum(len(w) for w in words) / max(1, len(words))

        return {
            "average_sentence_length": avg_sentence_length,
            "average_word_length": avg_word_length,
            "lexical_diversity": len(set(words)) / max(1, len(words))  # نسبت تعداد کلمات یکتا به کل کلمات
        }

    def extract_all_features(self, text: str) -> Dict[str, Any]:
        """
        اجرای همه عملیات استخراج ویژگی بر روی متن ورودی.
        :param text: متن ورودی
        :return: دیکشنری شامل تمام ویژگی‌های استخراج‌شده
        """
        return {
            "language": self.language,
            "syntax_features": self.extract_syntax_features(text),
            "semantic_features": self.extract_semantic_features(text),
            "text_complexity": self.extract_text_complexity(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    extractor = FeatureExtractor(language="fa")

    text_sample_en = "The smart whale processes language efficiently."
    text_sample_fa = "نهنگ هوشمند زبان را به‌طور بهینه پردازش می‌کند."
    text_sample_ru = "Умный кит эффективно обрабатывает язык."

    features_en = extractor.extract_all_features(text_sample_en)
    features_fa = extractor.extract_all_features(text_sample_fa)
    features_ru = extractor.extract_all_features(text_sample_ru)

    print("🔹 English Features:")
    print(features_en)

    print("\n🔹 Persian Features:")
    print(features_fa)

    print("\n🔹 Russian Features:")
    print(features_ru)
