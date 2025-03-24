import importlib
from typing import Dict, Any, Optional, List

class ContextProcessor:
    """
    این کلاس مسئول پردازش زمینه مکالمه و مدیریت ارتباط بین جملات است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای تحلیل عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None):
        """
        مقداردهی اولیه پردازشگر زمینه.
        :param language: زبان ورودی (در صورت `None`، زبان به‌طور خودکار شناسایی می‌شود)
        """
        self.language = language
        self.context_history = []  # حافظه کوتاه‌مدت مکالمه
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

    def update_context(self, text: str) -> None:
        """
        به‌روزرسانی حافظه مکالمه با جمله جدید.
        :param text: جمله ورودی جدید
        """
        self.context_history.append(text)
        if len(self.context_history) > 10:  # محدودیت تعداد جمله‌های ذخیره‌شده در حافظه کوتاه‌مدت
            self.context_history.pop(0)

    def analyze_context(self) -> Dict[str, Any]:
        """
        تحلیل کلی زمینه مکالمه با استفاده از معلم اختصاصی.
        :return: دیکشنری شامل تحلیل زمینه و الگوهای مکالمه
        """
        return self.language_processor.analyze_context(self.context_history)

    def process(self, text: str) -> Dict[str, Any]:
        """
        اجرای پردازش زمینه برای درک بهتر مکالمه.
        :param text: جمله ورودی
        :return: دیکشنری شامل تحلیل زمینه مکالمه
        """
        self.update_context(text)
        return {
            "language": self.language,
            "context_analysis": self.analyze_context(),
        }


# تست اولیه ماژول
if __name__ == "__main__":
    context_processor = ContextProcessor(language="fa")

    conversation = [
        "سلام، حال شما چطور است؟",
        "من دنبال یک مدل هوش مصنوعی برای تحلیل متن هستم.",
        "آیا مدل خاصی را پیشنهاد می‌کنید؟",
    ]

    for sentence in conversation:
        context_result = context_processor.process(sentence)
        print(f"\n🔹 Sentence: {sentence}")
        print("🔹 Context Analysis:")
        print(context_result)
