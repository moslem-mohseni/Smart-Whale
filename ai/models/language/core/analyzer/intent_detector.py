from typing import Dict, Any, Optional
from langdetect import detect
import importlib

class IntentDetector:
    """
    این کلاس مسئول تشخیص نیت کاربر از متن ورودی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای تحلیل عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه تشخیص نیت کاربر.
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
            return importlib.import_module("ai.models.language.adaptors.multilingual.intent_teacher").IntentTeacher()

    def detect_intent(self, text: str) -> Dict[str, Any]:
        """
        تشخیص نیت کاربر با استفاده از معلم اختصاصی هر زبان.
        :param text: جمله ورودی
        :return: دیکشنری شامل نیت تشخیص داده‌شده
        """
        return self.teacher_model.detect_intent(text)

    def analyze_intent(self, text: str) -> Dict[str, Any]:
        """
        تحلیل کلی نیت جمله شامل دسته‌بندی نوع درخواست و اطلاعات تکمیلی.
        اگر زبان، معلم اختصاصی داشته باشد، از `adaptors/` استفاده می‌شود.
        :param text: جمله ورودی
        :return: دیکشنری شامل نیت تشخیص داده‌شده
        """
        self.language = self._detect_language(text)
        self.teacher_model = self._load_teacher()

        return {
            "language": self.language,
            "intent_analysis": self.detect_intent(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    detector = IntentDetector()

    text_sample_en = "Can you help me find the best AI model?"
    text_sample_fa = "آیا می‌توانی به من کمک کنی بهترین مدل هوش مصنوعی را پیدا کنم؟"
    text_sample_ru = "Можешь помочь мне найти лучшую модель искусственного интеллекта?"

    analysis_result_en = detector.analyze_intent(text_sample_en)
    analysis_result_fa = detector.analyze_intent(text_sample_fa)
    analysis_result_ru = detector.analyze_intent(text_sample_ru)

    print("🔹 English Intent Analysis:")
    print(analysis_result_en)

    print("\n🔹 Persian Intent Analysis:")
    print(analysis_result_fa)

    print("\n🔹 Russian Intent Analysis:")
    print(analysis_result_ru)
