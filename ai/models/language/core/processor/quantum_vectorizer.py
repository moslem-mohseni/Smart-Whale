import importlib
import numpy as np
from typing import List, Dict, Any, Optional

class QuantumVectorizer:
    """
    این کلاس مسئول تبدیل متن به بردارهای عددی است.
    از معلم‌های اختصاصی برای زبان‌های خاص استفاده می‌کند و در صورت نبود معلم اختصاصی،
    از `mBERT` برای پردازش عمومی بهره می‌برد.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه بردارساز کوانتومی.
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

    def encode_text(self, text: str) -> np.ndarray:
        """
        تبدیل متن به بردار عددی با استفاده از مدل معلم.
        :param text: متن ورودی
        :return: بردار عددی نماینده متن
        """
        return self.language_processor.encode_text(text)

    def get_word_vectors(self, text: str) -> List[Dict[str, Any]]:
        """
        استخراج بردارهای عددی برای کلمات در متن.
        :param text: متن ورودی
        :return: لیستی از کلمات همراه با بردارهای آن‌ها
        """
        return self.language_processor.get_word_vectors(text)

    def vectorize(self, text: str) -> Dict[str, Any]:
        """
        اجرای فرآیند تبدیل متن به بردارهای عددی.
        :param text: متن ورودی
        :return: دیکشنری شامل بردار کلی متن و بردارهای کلمات
        """
        return {
            "language": self.language,
            "sentence_vector": self.encode_text(text).tolist(),
            "word_vectors": self.get_word_vectors(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    vectorizer = QuantumVectorizer(language="fa")

    text_sample_en = "The smart whale understands quantum language processing."
    text_sample_fa = "نهنگ هوشمند پردازش زبان کوانتومی را درک می‌کند."
    text_sample_ru = "Умный кит понимает квантовую обработку языка."

    vector_en = vectorizer.vectorize(text_sample_en)
    vector_fa = vectorizer.vectorize(text_sample_fa)
    vector_ru = vectorizer.vectorize(text_sample_ru)

    print("🔹 English Sentence Vector:")
    print(vector_en)

    print("\n🔹 Persian Sentence Vector:")
    print(vector_fa)

    print("\n🔹 Russian Sentence Vector:")
    print(vector_ru)
