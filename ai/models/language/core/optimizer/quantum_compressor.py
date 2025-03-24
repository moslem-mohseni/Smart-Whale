import importlib
import numpy as np
from typing import Dict, Any, Optional

class QuantumCompressor:
    """
    این کلاس مسئول فشرده‌سازی داده‌های زبانی به روش کوانتومی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای فشرده‌سازی عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None, compression_level: str = "standard"):
        """
        مقداردهی اولیه فشرده‌ساز کوانتومی.
        :param language: زبان ورودی (در صورت `None`، زبان به‌طور خودکار شناسایی می‌شود)
        :param compression_level: سطح فشرده‌سازی (`low`, `standard`, `high`)
        """
        self.language = language
        self.compression_level = compression_level
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

    def compress_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        فشرده‌سازی بردار ورودی با استفاده از الگوریتم‌های کوانتومی.
        :param vector: بردار ورودی
        :return: بردار فشرده‌شده
        """
        return self.language_processor.compress_vector(vector, self.compression_level)

    def compress_text(self, text: str) -> Dict[str, Any]:
        """
        فشرده‌سازی متن با تبدیل آن به بردار و اعمال روش‌های کوانتومی.
        :param text: متن ورودی
        :return: دیکشنری شامل داده‌های فشرده‌شده
        """
        vector = self.language_processor.quantum_transform(text)
        compressed_vector = self.compress_vector(vector)

        return {
            "language": self.language,
            "compression_level": self.compression_level,
            "compressed_data": compressed_vector.tolist(),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    compressor = QuantumCompressor(language="fa", compression_level="high")

    text_sample_en = "Quantum computing can revolutionize the way we process information."
    text_sample_fa = "رایانش کوانتومی می‌تواند شیوه پردازش اطلاعات را متحول کند."
    text_sample_ru = "Квантовые вычисления могут изменить способ обработки информации."

    compressed_en = compressor.compress_text(text_sample_en)
    compressed_fa = compressor.compress_text(text_sample_fa)
    compressed_ru = compressor.compress_text(text_sample_ru)

    print("🔹 English Compressed Data:")
    print(compressed_en)

    print("\n🔹 Persian Compressed Data:")
    print(compressed_fa)

    print("\n🔹 Russian Compressed Data:")
    print(compressed_ru)
