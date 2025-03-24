from typing import Dict, Any, Optional
from langdetect import detect
import importlib

class SemanticAnalyzer:
    """
    این کلاس مسئول تحلیل معنایی جملات ورودی است.
    از معلم اختصاصی برای هر زبان استفاده می‌کند و در صورت نبود معلم اختصاصی،
    از `mBERT` برای پردازش عمومی بهره می‌برد.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه تحلیلگر معنایی.
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
            return importlib.import_module("ai.models.language.adaptors.multilingual.semantic_teacher").SemanticTeacher()

    def extract_meaning(self, text: str) -> Dict[str, Any]:
        """
        استخراج معنا و مفهوم کلی جمله با استفاده از معلم اختصاصی هر زبان.
        :param text: جمله ورودی
        :return: دیکشنری شامل تحلیل معنایی جمله
        """
        return self.teacher_model.extract_meaning(text)

    def analyze_semantics(self, text: str) -> Dict[str, Any]:
        """
        تحلیل کلی معنایی جمله شامل استخراج مفهوم، روابط بین کلمات و مفاهیم کلیدی.
        اگر زبان، معلم اختصاصی داشته باشد، از `adaptors/` استفاده می‌شود.
        :param text: جمله ورودی
        :return: دیکشنری شامل تحلیل معنایی جمله
        """
        self.language = self._detect_language(text)
        self.teacher_model = self._load_teacher()

        return {
            "language": self.language,
            "semantic_analysis": self.extract_meaning(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    analyzer = SemanticAnalyzer()

    text_sample_en = "The smart whale understands complex data relationships."
    text_sample_fa = "نهنگ هوشمند روابط پیچیده داده‌ها را درک می‌کند."
    text_sample_ru = "Умный кит понимает сложные отношения данных."

    analysis_result_en = analyzer.analyze_semantics(text_sample_en)
    analysis_result_fa = analyzer.analyze_semantics(text_sample_fa)
    analysis_result_ru = analyzer.analyze_semantics(text_sample_ru)

    print("🔹 English Sentence Analysis:")
    print(analysis_result_en)

    print("\n🔹 Persian Sentence Analysis:")
    print(analysis_result_fa)

    print("\n🔹 Russian Sentence Analysis:")
    print(analysis_result_ru)
