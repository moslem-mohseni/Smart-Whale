import importlib
from typing import Dict, Any, Optional, List

class ResponseSelector:
    """
    این کلاس مسئول انتخاب بهترین پاسخ برای متن ورودی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای تحلیل عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه انتخابگر پاسخ.
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

    def generate_candidate_responses(self, text: str) -> List[Dict[str, Any]]:
        """
        تولید لیستی از پاسخ‌های ممکن بر اساس پردازش متن و زمینه.
        :param text: متن ورودی
        :return: لیستی از پاسخ‌های پیشنهادی
        """
        return self.language_processor.generate_candidate_responses(text)

    def rank_responses(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        امتیازدهی به پاسخ‌های تولیدشده و انتخاب پاسخ نهایی.
        :param responses: لیست پاسخ‌های ممکن
        :return: بهترین پاسخ با بیشترین امتیاز
        """
        sorted_responses = sorted(responses, key=lambda x: x["score"], reverse=True)
        return sorted_responses[0] if sorted_responses else {"response": "متأسفم، پاسخ مناسبی پیدا نشد."}

    def select_response(self, text: str) -> Dict[str, Any]:
        """
        پردازش کامل انتخاب پاسخ شامل تولید و امتیازدهی به پاسخ‌ها.
        :param text: متن ورودی
        :return: دیکشنری شامل بهترین پاسخ انتخاب‌شده
        """
        candidate_responses = self.generate_candidate_responses(text)
        best_response = self.rank_responses(candidate_responses)
        return {
            "language": self.language,
            "selected_response": best_response,
        }


# تست اولیه ماژول
if __name__ == "__main__":
    response_selector = ResponseSelector(language="fa")

    text_sample_en = "What is the best AI model for text analysis?"
    text_sample_fa = "بهترین مدل هوش مصنوعی برای تحلیل متن چیست؟"
    text_sample_ru = "Какая лучшая модель ИИ для анализа текста?"

    response_en = response_selector.select_response(text_sample_en)
    response_fa = response_selector.select_response(text_sample_fa)
    response_ru = response_selector.select_response(text_sample_ru)

    print("🔹 English Response:")
    print(response_en)

    print("\n🔹 Persian Response:")
    print(response_fa)

    print("\n🔹 Russian Response:")
    print(response_ru)
