import re
from hazm import Normalizer


class TextCleaner:
    """
    کلاس پاک‌سازی و نرمال‌سازی متن از نویزهای متنی
    """

    def __init__(self):
        self.normalizer = Normalizer()

    def clean_text(self, text):
        """
        پاک‌سازی متن از نویزهای متنی و نرمال‌سازی آن
        :param text: متن ورودی
        :return: متن پردازش‌شده
        """
        text = re.sub(r"\s+", " ", text).strip()  # حذف فاصله‌های اضافی
        text = re.sub(r"http\S+|www\S+", "", text)  # حذف لینک‌ها
        text = re.sub(r"@\w+", "", text)  # حذف منشن‌ها (@username)
        text = re.sub(r"#\w+", "", text)  # حذف هشتگ‌ها (#Hashtag)
        text = re.sub(r"[^\w\s]", "", text)  # حذف کاراکترهای خاص
        text = self.normalizer.normalize(text)  # نرمال‌سازی متن فارسی
        return text


# ✅ تست داخلی برای بررسی صحت عملکرد پاک‌سازی متن
if __name__ == "__main__":
    cleaner = TextCleaner()

    sample_text = "این یک #تست است! لطفاً @user به این لینک مراجعه کنید: https://example.com 😃"
    cleaned_text = cleaner.clean_text(sample_text)

    print("✅ متن قبل از پاک‌سازی:")
    print(sample_text)

    print("\n✅ متن بعد از پاک‌سازی:")
    print(cleaned_text)
