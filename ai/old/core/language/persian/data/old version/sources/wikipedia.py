import re
import wikipediaapi as wikiapi
import json
import os
from hazm import Normalizer


class WikipediaExtractor:
    def __init__(self, language="fa", max_length=5000, cache_enabled=True):
        """
        کلاس استخراج متن از ویکی‌پدیا

        :param language: زبان ویکی‌پدیا (پیش‌فرض: فارسی)
        :param max_length: حداکثر طول متن (برای کنترل حجم داده‌های پردازشی)
        :param cache_enabled: فعال‌سازی کش برای کاهش درخواست‌های اضافی
        """
        self.user_agent = "SmartWhaleBot/1.0"
        self.language = language
        self.max_length = max_length
        self.cache_enabled = cache_enabled
        self.cache_dir = "cache"
        self.normalizer = Normalizer()
        self.wiki = wikiapi.Wikipedia(language=language, user_agent=self.user_agent)

        if cache_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_page_text(self, title):
        """
        دریافت متن یک صفحه از ویکی‌پدیا

        :param title: عنوان صفحه ویکی‌پدیا
        :return: متن تمیز و پردازش‌شده
        """
        # بررسی کش برای جلوگیری از درخواست‌های اضافی
        if self.cache_enabled:
            cached_text = self._load_from_cache(title)
            if cached_text:
                return cached_text

        # دریافت متن از ویکی‌پدیا
        try:
            raw_text = self._fetch_wikipedia_page(title)
            cleaned_text = self._clean_text(raw_text)

            # ذخیره در کش
            if self.cache_enabled:
                self._save_to_cache(title, cleaned_text)

            return cleaned_text[:self.max_length]

        except ValueError as e:
            print(f"⚠ خطا: {e}")
            return None

    def _fetch_wikipedia_page(self, title):
        """
        دریافت متن خام از ویکی‌پدیا

        :param title: عنوان صفحه ویکی‌پدیا
        :return: متن خام صفحه
        """
        page = self.wiki.page(title)
        if not page.exists():
            raise ValueError(f"⚠ صفحه '{title}' در ویکی‌پدیا یافت نشد.")
        return page.text

    def _clean_text(self, text):
        """
        حذف بخش‌های اضافی و تمیز کردن متن ویکی‌پدیا

        :param text: متن خام ویکی‌پدیا
        :return: متن پردازش‌شده
        """
        text = re.sub(r"\[\d+\]", "", text)  # حذف شماره‌های منبع ([1], [2] و ...)
        text = re.sub(r"https?://\S+", "", text)  # حذف لینک‌ها
        text = re.sub(r"\s+", " ", text).strip()  # حذف فاصله‌های اضافی
        text = self.normalizer.normalize(text)  # نرمال‌سازی متن
        return text

    def _load_from_cache(self, title):
        """
        بارگذاری صفحه از کش در صورت موجود بودن

        :param title: عنوان صفحه ویکی‌پدیا
        :return: متن کش‌شده (در صورت موجود بودن) یا None
        """
        cache_path = os.path.join(self.cache_dir, f"{title}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as file:
                cached_data = json.load(file)
                return cached_data.get("text")
        return None

    def _save_to_cache(self, title, text):
        """
        ذخیره متن صفحه در کش برای کاهش درخواست‌های اضافی

        :param title: عنوان صفحه ویکی‌پدیا
        :param text: متن پردازش‌شده برای ذخیره در کش
        """
        cache_path = os.path.join(self.cache_dir, f"{title}.json")
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump({"title": title, "text": text}, file, ensure_ascii=False, indent=4)


# مثال استفاده
if __name__ == "__main__":
    wiki_extractor = WikipediaExtractor(language="fa", max_length=3000, cache_enabled=True)

    try:
        page_title = "هوش مصنوعی"
        content = wiki_extractor.get_page_text(page_title)
        print(f"✅ محتوای صفحه '{page_title}':\n", content[:500], "...")  # نمایش فقط 500 کاراکتر اول
    except ValueError as e:
        print(e)
