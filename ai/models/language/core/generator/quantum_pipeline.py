import importlib
import numpy as np
from typing import Dict, Any, Optional

class QuantumPipeline:
    """
    این کلاس مسئول پردازش و تولید پاسخ‌های زبانی به روش کوانتومی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای پردازش عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه سیستم پردازش کوانتومی.
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

    def quantum_transform(self, text: str) -> np.ndarray:
        """
        تبدیل متن به بردار کوانتومی برای پردازش دقیق‌تر.
        :param text: متن ورودی
        :return: بردار کوانتومی نماینده متن
        """
        return self.language_processor.quantum_transform(text)

    def generate_quantum_response(self, text: str) -> Dict[str, Any]:
        """
        تولید پاسخ با استفاده از پردازش کوانتومی.
        :param text: متن ورودی
        :return: پاسخ تولیدشده به روش کوانتومی
        """
        return self.language_processor.generate_quantum_response(text)

    def process(self, text: str) -> Dict[str, Any]:
        """
        پردازش کامل تولید پاسخ مبتنی بر پردازش کوانتومی.
        :param text: متن ورودی
        :return: دیکشنری شامل پاسخ تولیدشده
        """
        return {
            "language": self.language,
            "quantum_vector": self.quantum_transform(text).tolist(),
            "generated_response": self.generate_quantum_response(text),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    quantum_pipeline = QuantumPipeline(language="fa")

    text_sample_en = "Explain quantum computing in simple terms."
    text_sample_fa = "رایانش کوانتومی را به زبان ساده توضیح بده."
    text_sample_ru = "Объясните квантовые вычисления простыми словами."

    response_en = quantum_pipeline.process(text_sample_en)
    response_fa = quantum_pipeline.process(text_sample_fa)
    response_ru = quantum_pipeline.process(text_sample_ru)

    print("🔹 English Quantum Response:")
    print(response_en)

    print("\n🔹 Persian Quantum Response:")
    print(response_fa)

    print("\n🔹 Russian Quantum Response:")
    print(response_ru)
