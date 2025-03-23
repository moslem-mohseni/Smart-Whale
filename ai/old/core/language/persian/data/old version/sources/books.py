import os
import re
import json
import pymupdf as fitz
from docx import Document
from hazm import Normalizer


class BookProcessor:
    def __init__(self, cache_enabled=True, max_length=500000, accepted_formats=None):
        """
        کلاس پردازش کتاب‌ها و مقالات فارسی (PDF, TXT, DOCX)

        :param cache_enabled: فعال‌سازی کش برای جلوگیری از پردازش مجدد فایل‌های قبلی
        :param max_length: حداکثر طول متن خروجی برای کنترل حجم پردازش
        :param accepted_formats: فرمت‌های مجاز برای پردازش (پیش‌فرض: PDF, TXT, DOCX)
        """
        self.cache_enabled = cache_enabled
        self.max_length = max_length
        self.cache_dir = "cache/books"
        self.normalizer = Normalizer()
        self.accepted_formats = accepted_formats or {".pdf", ".txt", ".docx"}

        if cache_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def process_book(self, file_path):
        """
        پردازش یک کتاب و استخراج محتوای آن

        :param file_path: مسیر فایل کتاب
        :return: متن پردازش‌شده‌ی کتاب
        """
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

        # خواندن و پردازش متن
        if file_extension == ".pdf":
            text = self._extract_text_from_pdf(file_path)
        elif file_extension == ".txt":
            text = self._extract_text_from_txt(file_path)
        elif file_extension == ".docx":
            text = self._extract_text_from_docx(file_path)
        else:
            raise ValueError("⚠ فرمت فایل پشتیبانی نمی‌شود.")

        cleaned_text = self._clean_text(text)

        # ذخیره در کش
        if self.cache_enabled:
            self._save_to_cache(file_path, cleaned_text)

        return cleaned_text[:self.max_length]

    @staticmethod
    def _extract_text_from_pdf(file_path):
        """
        استخراج متن از فایل PDF

        :param file_path: مسیر فایل PDF
        :return: متن خام استخراج‌شده
        """
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") if hasattr(page, "get_text") else page.getText("text") for page in doc])
        return text

    @staticmethod
    def _extract_text_from_txt(file_path):
        """
        خواندن متن از فایل TXT

        :param file_path: مسیر فایل TXT
        :return: متن خام
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _extract_text_from_docx(file_path):
        """
        استخراج متن از فایل DOCX

        :param file_path: مسیر فایل DOCX
        :return: متن خام استخراج‌شده
        """
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    def _clean_text(self, text):
        """
        تمیز کردن و نرمال‌سازی متن کتاب

        :param text: متن خام
        :return: متن پردازش‌شده
        """
        text = re.sub(r"\s+", " ", text).strip()  # حذف فاصله‌های اضافی
        text = re.sub(r"https?://\S+", "", text)  # حذف لینک‌ها
        text = re.sub(r"\[\d+\]", "", text)  # حذف منابع مانند [1], [2]
        text = self.normalizer.normalize(text)  # نرمال‌سازی متن فارسی
        return text

    def _load_from_cache(self, file_path):
        """
        بارگذاری متن از کش در صورت موجود بودن

        :param file_path: مسیر فایل
        :return: متن کش‌شده یا None
        """
        cache_path = os.path.join(self.cache_dir, f"{os.path.basename(file_path)}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as file:
                cached_data = json.load(file)
                return cached_data.get("text")
        return None

    def _save_to_cache(self, file_path, text):
        """
        ذخیره متن پردازش‌شده در کش

        :param file_path: مسیر فایل
        :param text: متن پردازش‌شده برای ذخیره در کش
        """
        cache_path = os.path.join(self.cache_dir, f"{os.path.basename(file_path)}.json")
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump({"file": file_path, "text": text}, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    book_processor = BookProcessor(cache_enabled=False)

    # مسیر فایل‌های تستی
    pdf_file = "example.pdf"  # اگر فایل PDF داری
    txt_file = "example.txt"  # اگر فایل TXT داری
    docx_file = "example.docx"  # اگر فایل DOCX داری

    try:
        txt_text = book_processor.process_book(txt_file)
        print(f"✅ پردازش TXT موفق: {txt_text[:300]} ...")  # نمایش 300 کاراکتر اول

        docx_text = book_processor.process_book(docx_file)
        print(f"✅ پردازش DOCX موفق: {docx_text[:300]} ...")  # نمایش 300 کاراکتر اول

        pdf_text = book_processor.process_book(pdf_file)
        print(f"✅ پردازش PDF موفق: {pdf_text[:300]} ...")  # نمایش 300 کاراکتر اول

    except Exception as e:
        print(f"❌ خطا در پردازش: {e}")
