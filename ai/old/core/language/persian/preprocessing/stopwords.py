import re
from hazm import WordTokenizer


class PersianStopWords:
    def __init__(self, custom_stopwords_path=None):
        """
        کلاس حذف کلمات توقف برای پردازش زبان فارسی.

        :param custom_stopwords_path: مسیر فایل کلمات توقف سفارشی (در صورت وجود)
        """
        self.tokenizer = WordTokenizer()
        self.stopwords = self._load_stopwords(custom_stopwords_path) if custom_stopwords_path else self._default_stopwords()

    def remove_stopwords(self, text):
        """
        حذف کلمات توقف از یک متن

        :param text: متن فارسی ورودی
        :return: متن بدون کلمات توقف
        """
        words = self.tokenizer.tokenize(text)
        filtered_words = [word for word in words if word not in self.stopwords]
        return " ".join(filtered_words)

    def _load_stopwords(self, stopwords_path):
        """
        بارگذاری لیست کلمات توقف از فایل

        :param stopwords_path: مسیر فایل کلمات توقف
        :return: مجموعه‌ای از کلمات توقف
        """
        try:
            with open(stopwords_path, "r", encoding="utf-8") as file:
                return set(file.read().splitlines())
        except FileNotFoundError:
            print("⚠ فایل کلمات توقف پیدا نشد، از لیست پیش‌فرض استفاده می‌شود.")
            return self._default_stopwords()

    def _default_stopwords(self):
        """
        لیست پیش‌فرض کلمات توقف فارسی

        :return: مجموعه‌ای از کلمات توقف
        """
        return {
            "و", "در", "به", "از", "که", "را", "این", "آن", "برای", "یک", "با", "بر",
            "نیز", "هم", "ولی", "تا", "نه", "چون", "اگر", "اما", "است", "بود", "می‌شود",
            "شود", "کرد", "شد", "دارد", "همه", "هر", "باید", "نیست", "کنید", "دهد"
        }


# مثال استفاده
if __name__ == "__main__":
    stopwords_remover = PersianStopWords()
    text = "این یک متن آزمایشی است که باید پردازش شود و کلمات اضافی حذف شوند."
    print(stopwords_remover.remove_stopwords(text))
