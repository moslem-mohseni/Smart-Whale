import re
import importlib
from typing import Optional


class TextNormalizer:
    """
    این کلاس مسئول نرمال‌سازی متن ورودی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای تحلیل عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه نرمال‌ساز متن.
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

    def remove_noise(self, text: str) -> str:
        """
        حذف نویزهای متنی از جمله کاراکترهای نامناسب، فاصله‌های اضافی و نشانه‌گذاری غیرضروری.
        :param text: متن ورودی
        :return: متن پردازش‌شده
        """
        text = re.sub(r"[^A-Za-z0-9\u0600-\u06FF\s]", "", text)  # حذف کاراکترهای غیرمجاز (شامل فارسی، انگلیسی و اعداد)
        text = re.sub(r"\s+", " ", text).strip()  # حذف فاصله‌های اضافی
        return text

    def normalize_text(self, text: str) -> str:
        """
        اجرای عملیات نرمال‌سازی متن از جمله حذف نویز، اصلاح علائم و یکسان‌سازی فرمت متن.
        اگر زبان، معلم اختصاصی داشته باشد، از `adaptors/` استفاده می‌شود.
        :param text: متن ورودی
        :return: متن نرمال‌شده
        """
        text = self.remove_noise(text)
        return self.language_processor.normalize_text(text)

    def process(self, text: str) -> str:
        """
        پردازش کامل متن شامل همه مراحل نرمال‌سازی.
        :param text: متن ورودی
        :return: متن پردازش‌شده نهایی
        """
        return self.normalize_text(text)


# تست اولیه ماژول
if __name__ == "__main__":
    normalizer = TextNormalizer(language="fa")

    text_sample_en = " Hello,   world!!! This is a     test message...  "
    text_sample_fa = "سلام!!!   دنیا  !!! این    یک    پیام آزمایشی است... "
    text_sample_ru = "Привет!!   мир!  Это тестовое сообщение..."

    normalized_en = normalizer.process(text_sample_en)
    normalized_fa = normalizer.process(text_sample_fa)
    normalized_ru = normalizer.process(text_sample_ru)

    print("🔹 English Normalized Text:")
    print(normalized_en)

    print("\n🔹 Persian Normalized Text:")
    print(normalized_fa)

    print("\n🔹 Russian Normalized Text:")
    print(normalized_ru)
