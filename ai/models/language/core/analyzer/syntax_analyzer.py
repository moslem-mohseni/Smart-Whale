from typing import List, Dict, Any, Optional
from langdetect import detect
import importlib

class SyntaxAnalyzer:
    """
    این کلاس مسئول تحلیل نحوی جملات ورودی است.
    این نسخه شامل مکانیسمی برای استفاده از معلم‌های اختصاصی برای هر زبان،
    و همچنین پشتیبانی از زبان‌های بدون معلم اختصاصی با مدل عمومی `mBERT` است.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه تحلیلگر نحوی.
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
            # بررسی اینکه آیا زبان ورودی ماژول اختصاصی دارد یا نه
            module_path = f"ai.models.language.adaptors.{self.language}.language_processor"
            return importlib.import_module(module_path).LanguageProcessor()
        except ModuleNotFoundError:
            # اگر زبان معلم اختصاصی ندارد، از `mBERT` برای تحلیل عمومی استفاده می‌کنیم
            return importlib.import_module("ai.models.language.adaptors.multilingual.syntax_teacher").SyntaxTeacher()

    def pos_tagging(self, text: str) -> List[Dict[str, Any]]:
        """
        تحلیل نحوی و برچسب‌گذاری کلمات بر اساس نقش گرامری (POS Tagging).
        معلم اختصاصی در صورت موجود بودن استفاده می‌شود.
        :param text: جمله ورودی
        :return: لیستی از کلمات به همراه برچسب نحوی آن‌ها
        """
        return self.teacher_model.pos_tagging(text)

    def dependency_parsing(self, text: str) -> List[Dict[str, Any]]:
        """
        تجزیه نحوی جمله و نمایش وابستگی‌های نحوی.
        در صورت وجود معلم اختصاصی، از آن استفاده می‌شود.
        :param text: جمله ورودی
        :return: لیستی شامل وابستگی‌های نحوی کلمات
        """
        return self.teacher_model.dependency_parsing(text)

    def generate_parse_tree(self, text: str) -> str:
        """
        تولید نمایش متنی از درخت نحوی جمله.
        در صورت وجود پردازشگر اختصاصی، از آن استفاده می‌شود.
        :param text: جمله ورودی
        :return: رشته متنی شامل نمایش ساختار درخت نحوی
        """
        return self.teacher_model.generate_parse_tree(text)

    def analyze_syntax(self, text: str) -> Dict[str, Any]:
        """
        تحلیل کلی نحوی جمله شامل برچسب‌گذاری، تجزیه نحوی و تولید درخت نحوی.
        اگر زبان، معلم اختصاصی داشته باشد، از `adaptors/` استفاده می‌شود.
        :param text: جمله ورودی
        :return: دیکشنری شامل تحلیل نحوی جمله
        """
        self.language = self._detect_language(text)
        self.teacher_model = self._load_teacher()

        return {
            "language": self.language,
            "pos_tags": self.pos_tagging(text),
            "dependency_tree": self.dependency_parsing(text),
            "parse_tree": self.generate_parse_tree(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    analyzer = SyntaxAnalyzer()

    text_sample_en = "The smart whale processes data efficiently."
    text_sample_fa = "نهنگ هوشمند داده‌ها را به‌طور بهینه پردازش می‌کند."
    text_sample_ru = "Умный кит эффективно обрабатывает данные."

    analysis_result_en = analyzer.analyze_syntax(text_sample_en)
    analysis_result_fa = analyzer.analyze_syntax(text_sample_fa)
    analysis_result_ru = analyzer.analyze_syntax(text_sample_ru)

    print("🔹 English Sentence Analysis:")
    print(analysis_result_en)

    print("\n🔹 Persian Sentence Analysis:")
    print(analysis_result_fa)

    print("\n🔹 Russian Sentence Analysis:")
    print(analysis_result_ru)
