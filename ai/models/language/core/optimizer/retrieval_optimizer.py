import importlib
import numpy as np
from typing import Dict, Any, Optional, List

class RetrievalOptimizer:
    """
    این کلاس مسئول بهینه‌سازی فرآیند بازیابی اطلاعات زبانی است.
    پردازش توسط معلم اختصاصی هر زبان انجام می‌شود و در صورت نبود معلم اختصاصی،
    از `mBERT` برای بازیابی عمومی استفاده می‌شود.
    """

    def __init__(self, language: Optional[str] = None, retrieval_method: str = "semantic"):
        """
        مقداردهی اولیه بهینه‌ساز بازیابی اطلاعات.
        :param language: زبان ورودی (در صورت `None`، زبان به‌طور خودکار شناسایی می‌شود)
        :param retrieval_method: روش بازیابی (`keyword`, `semantic`, `hybrid`)
        """
        self.language = language
        self.retrieval_method = retrieval_method
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

    def keyword_search(self, query: str, documents: List[str]) -> List[str]:
        """
        بازیابی اطلاعات بر اساس جستجوی کلمات کلیدی.
        :param query: متن جستجو
        :param documents: لیستی از اسناد برای جستجو
        :return: لیستی از اسناد مرتبط
        """
        return self.language_processor.keyword_search(query, documents)

    def semantic_search(self, query: str, documents: List[str]) -> List[str]:
        """
        بازیابی اطلاعات بر اساس جستجوی معنایی.
        :param query: متن جستجو
        :param documents: لیستی از اسناد برای جستجو
        :return: لیستی از اسناد مرتبط
        """
        return self.language_processor.semantic_search(query, documents)

    def hybrid_search(self, query: str, documents: List[str]) -> List[str]:
        """
        ترکیب جستجوی کلمات کلیدی و جستجوی معنایی.
        :param query: متن جستجو
        :param documents: لیستی از اسناد برای جستجو
        :return: لیستی از اسناد مرتبط
        """
        keyword_results = self.keyword_search(query, documents)
        semantic_results = self.semantic_search(query, documents)
        return list(set(keyword_results + semantic_results))  # ادغام نتایج جستجو

    def retrieve_documents(self, query: str, documents: List[str]) -> Dict[str, Any]:
        """
        اجرای فرآیند بازیابی اطلاعات با روش مشخص‌شده.
        :param query: متن جستجو
        :param documents: لیستی از اسناد برای جستجو
        :return: دیکشنری شامل اسناد بازیابی‌شده
        """
        if self.retrieval_method == "keyword":
            results = self.keyword_search(query, documents)
        elif self.retrieval_method == "semantic":
            results = self.semantic_search(query, documents)
        else:  # hybrid
            results = self.hybrid_search(query, documents)

        return {
            "language": self.language,
            "retrieval_method": self.retrieval_method,
            "retrieved_documents": results,
        }


# تست اولیه ماژول
if __name__ == "__main__":
    retrieval_optimizer = RetrievalOptimizer(language="fa", retrieval_method="hybrid")

    query_en = "Best practices in artificial intelligence"
    query_fa = "بهترین روش‌های هوش مصنوعی"
    query_ru = "Лучшие практики искусственного интеллекта"

    documents = [
        "Artificial intelligence is revolutionizing the world.",
        "بهترین روش‌های یادگیری ماشین در پردازش زبان طبیعی",
        "Современные методы машинного обучения",
        "AI best practices include data preprocessing and model optimization.",
        "پردازش زبان طبیعی یکی از مهم‌ترین شاخه‌های هوش مصنوعی است."
    ]

    results_en = retrieval_optimizer.retrieve_documents(query_en, documents)
    results_fa = retrieval_optimizer.retrieve_documents(query_fa, documents)
    results_ru = retrieval_optimizer.retrieve_documents(query_ru, documents)

    print("🔹 English Retrieval Results:")
    print(results_en)

    print("\n🔹 Persian Retrieval Results:")
    print(results_fa)

    print("\n🔹 Russian Retrieval Results:")
    print(results_ru)
