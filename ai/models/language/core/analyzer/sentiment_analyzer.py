from typing import Dict, Any, Optional
from langdetect import detect
import importlib

class SentimentAnalyzer:
    """
    این کلاس مسئول تحلیل احساسات متن ورودی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای تحلیل عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه تحلیلگر احساسات.
        :param language: زبان ورودی (در صورت `None`، زبان به‌طور خودکار شناسایی می‌شود)
        """
        self.language = language
        self.teacher_model = self._load_teacher()

    def _detect_language(self, text: str) -> str:
        """
        تشخیص زبان متن ورودی در صورت عدم تعیین زبان در مقداردهی اولیه.
        :param text: جمله ورودی
        :return: زبان شناسایی‌شده
        """
        if not self.language:
            try:
                detected_lang = detect(text)
                return detected_lang
            except:
                return "unknown"
        return self.language

    def _load_teacher(self):
        """
        بررسی و بارگذاری معلم اختصاصی در صورت وجود.
        :return: ماژول معلم اختصاصی یا معلم عمومی (`mBERT`) در صورت عدم وجود
        """
        try:
            module_path = f"ai.models.language.adaptors.{self.language}.language_processor"
            return importlib.import_module(module_path).LanguageProcessor()
        except ModuleNotFoundError:
            # استفاده از `mBERT` برای زبان‌هایی که معلم اختصاصی ندارند
            return importlib.import_module("ai.models.language.adaptors.multilingual.sentiment_teacher").SentimentTeacher()

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        تحلیل احساسات متن شامل تشخیص نوع و شدت احساس.
        اگر زبان، معلم اختصاصی داشته باشد، از `adaptors/` استفاده می‌شود.
        :param text: جمله ورودی
        :return: دیکشنری شامل نتیجه تحلیل احساسات
        """
        return self.teacher_model.analyze_sentiment(text)

    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """
        اجرای تحلیل احساسات با تشخیص خودکار زبان.
        :param text: جمله ورودی
        :return: دیکشنری شامل نتیجه تحلیل احساسات
        """
        self.language = self._detect_language(text)
        self.teacher_model = self._load_teacher()

        return {
            "language": self.language,
            "sentiment_analysis": self.analyze_sentiment(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    text_sample_en = "I love this product! It's absolutely amazing!"
    text_sample_fa = "من این محصول را دوست دارم! فوق‌العاده است!"
    text_sample_ru = "Мне очень нравится этот продукт! Он потрясающий!"

    analysis_result_en = analyzer.get_sentiment(text_sample_en)
    analysis_result_fa = analyzer.get_sentiment(text_sample_fa)
    analysis_result_ru = analyzer.get_sentiment(text_sample_ru)

    print("🔹 English Sentiment Analysis:")
    print(analysis_result_en)

    print("\n🔹 Persian Sentiment Analysis:")
    print(analysis_result_fa)

    print("\n🔹 Russian Sentiment Analysis:")
    print(analysis_result_ru)
