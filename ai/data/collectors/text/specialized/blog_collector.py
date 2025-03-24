import os
import json
import pymupdf as fitz
from docx import Document


class BooksScraper:
    """
    کلاس جمع‌آوری و استخراج داده از کتاب‌ها و مقالات فارسی (PDF, TXT, DOCX)
    """

    def __init__(self, cache_enabled=True, max_length=500000, accepted_formats=None):
        self.cache_enabled = cache_enabled
        self.max_length = max_length
        self.cache_dir = "cache/books"
        self.accepted_formats = accepted_formats or {".pdf", ".txt", ".docx"}

        if cache_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def extract_text(self, file_path):
        """استخراج متن از کتاب‌ها"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"⚠ فایل '{file_path}' یافت نشد.")

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in self.accepted_formats:
            raise ValueError(f"⚠ فرمت '{file_extension}' پشتیبانی نمی‌شود.")

        # بررسی کش
        if self.cache_enabled:
            cached_text = self._load_from_cache(file_path)
            if cached_text:
                return cached_text

        # استخراج متن
        text = self._extract_text(file_path, file_extension)

        if self.cache_enabled:
            self._save_to_cache(file_path, text)

        return text[:self.max_length]

    def _extract_text(self, file_path, file_extension):
        """استخراج متن از فایل‌های پشتیبانی‌شده"""
        if file_extension == ".pdf":
            return self._extract_text_from_pdf(file_path)
        elif file_extension == ".txt":
            return self._extract_text_from_txt(file_path)
        elif file_extension == ".docx":
            return self._extract_text_from_docx(file_path)
        else:
            raise ValueError("⚠ فرمت فایل پشتیبانی نمی‌شود.")

    @staticmethod
    def _extract_text_from_pdf(file_path):
        doc = fitz.open(file_path)
        return "\n".join([page.get_text("text") for page in doc])

    @staticmethod
    def _extract_text_from_txt(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _extract_text_from_docx(file_path):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _load_from_cache(self, file_path):
        cache_path = os.path.join(self.cache_dir, f"{os.path.basename(file_path)}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as file:
                cached_data = json.load(file)
                return cached_data.get("text")
        return None

    def _save_to_cache(self, file_path, text):
        cache_path = os.path.join(self.cache_dir, f"{os.path.basename(file_path)}.json")
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump({"file": file_path, "text": text}, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    scraper = BooksScraper(cache_enabled=False)
    sample_file = "example.pdf"
    try:
        extracted_text = scraper.extract_text(sample_file)
        print(f"✅ استخراج موفق: {extracted_text[:300]} ...")
    except Exception as e:
        print(f"❌ خطا در پردازش: {e}")
