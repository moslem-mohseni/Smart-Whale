from typing import Dict, Any, Optional, List
from langdetect import detect
import importlib

class EntityRecognizer:
    """
    این کلاس مسئول شناسایی موجودیت‌های نامدار در متن ورودی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای تحلیل عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه تشخیص موجودیت‌ها.
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
            return importlib.import_module("ai.models.language.adaptors.multilingual.entity_teacher").EntityTeacher()

    def recognize_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        شناسایی موجودیت‌های نامدار در جمله با استفاده از معلم اختصاصی هر زبان.
        :param text: جمله ورودی
        :return: لیستی از موجودیت‌های شناسایی‌شده
        """
        return self.teacher_model.recognize_entities(text)

    def analyze_entities(self, text: str) -> Dict[str, Any]:
        """
        تحلیل کلی موجودیت‌های جمله شامل نام اشخاص، مکان‌ها، سازمان‌ها و غیره.
        اگر زبان، معلم اختصاصی داشته باشد، از `adaptors/` استفاده می‌شود.
        :param text: جمله ورودی
        :return: دیکشنری شامل لیست موجودیت‌های شناسایی‌شده
        """
        self.language = self._detect_language(text)
        self.teacher_model = self._load_teacher()

        return {
            "language": self.language,
            "named_entities": self.recognize_entities(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    recognizer = EntityRecognizer()

    text_sample_en = "Elon Musk is the CEO of Tesla and was born in South Africa."
    text_sample_fa = "الون ماسک مدیرعامل تسلا است و در آفریقای جنوبی به دنیا آمده است."
    text_sample_ru = "Илон Маск - генеральный директор Tesla, родился в Южной Африке."

    analysis_result_en = recognizer.analyze_entities(text_sample_en)
    analysis_result_fa = recognizer.analyze_entities(text_sample_fa)
    analysis_result_ru = recognizer.analyze_entities(text_sample_ru)

    print("🔹 English Entity Recognition:")
    print(analysis_result_en)

    print("\n🔹 Persian Entity Recognition:")
    print(analysis_result_fa)

    print("\n🔹 Russian Entity Recognition:")
    print(analysis_result_ru)
