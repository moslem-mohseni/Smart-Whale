import re
from hazm import WordTokenizer


class PersianTokenizer:
    def __init__(self, separate_punctuation=True, normalize_text=True):
        """
        کلاس توکن‌سازی برای زبان فارسی

        :param separate_punctuation: اگر True باشد، علائم نگارشی به‌عنوان توکن مستقل در نظر گرفته می‌شوند.
        :param normalize_text: اگر True باشد، متن قبل از توکن‌سازی نرمال‌سازی می‌شود.
        """
        self.tokenizer = WordTokenizer(separate_emoji=True)
        self.separate_punctuation = separate_punctuation
        self.normalize_text = normalize_text

    def tokenize(self, text):
        """
        توکن‌سازی یک متن ورودی

        :param text: متن فارسی ورودی
        :return: لیست توکن‌ها
        """
        if self.normalize_text:
            text = self._normalize(text)

        tokens = self.tokenizer.tokenize(text)

        if self.separate_punctuation:
            tokens = self._split_punctuation(tokens)

        return tokens

    def _normalize(self, text):
        """
        نرمال‌سازی اولیه متن (تبدیل اعداد عربی به فارسی، حذف اسپیس‌های اضافی و ...)

        :param text: متن ورودی
        :return: متن نرمال‌شده
        """
        text = re.sub(r"[ي]", "ی", text)  # تبدیل ی عربی به فارسی
        text = re.sub(r"[ك]", "ک", text)  # تبدیل ک عربی به فارسی
        text = re.sub(r"\s+", " ", text).strip()  # حذف فاصله‌های اضافی
        text = re.sub(r"[۰-۹]", lambda x: str(int(x.group())), text)  # تبدیل اعداد فارسی به انگلیسی
        return text

    def _split_punctuation(self, tokens):
        """
        جداسازی علائم نگارشی به‌عنوان توکن مستقل

        :param tokens: لیست توکن‌ها
        :return: لیست توکن‌های پردازش‌شده
        """
        processed_tokens = []
        for token in tokens:
            split_tokens = re.findall(r"[\w]+|[^\s\w]", token)
            processed_tokens.extend(split_tokens)
        return processed_tokens


# مثال استفاده
if __name__ == "__main__":
    tokenizer = PersianTokenizer()
    text = "این یک جمله‌ی تستی است! آیا به‌درستی توکن‌سازی می‌شود؟"
    print(tokenizer.tokenize(text))
