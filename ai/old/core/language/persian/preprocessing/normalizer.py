import re
from hazm import Normalizer


class PersianNormalizer:
    def __init__(self, remove_extra_spaces=True, fix_half_space=True, convert_numbers=True):
        """
        کلاس نرمال‌ساز متن فارسی

        :param remove_extra_spaces: اگر True باشد، فاصله‌های اضافی حذف می‌شوند.
        :param fix_half_space: اگر True باشد، نیم‌فاصله‌های نادرست اصلاح می‌شوند.
        :param convert_numbers: اگر True باشد، اعداد عربی و فارسی به انگلیسی تبدیل می‌شوند.
        """
        self.normalizer = Normalizer()
        self.remove_extra_spaces = remove_extra_spaces
        self.fix_half_space = fix_half_space
        self.convert_numbers = convert_numbers

    def normalize(self, text):
        """
        نرمال‌سازی متن ورودی

        :param text: متن فارسی
        :return: متن نرمال‌شده
        """
        # تبدیل حروف عربی به فارسی
        text = self._convert_arabic_chars(text)

        # نرمال‌سازی با hazm
        text = self.normalizer.normalize(text)

        # تبدیل اعداد عربی/فارسی به انگلیسی
        if self.convert_numbers:
            text = self._convert_numbers(text)

        # حذف فاصله‌های اضافی
        if self.remove_extra_spaces:
            text = self._remove_extra_spaces(text)

        # اصلاح نیم‌فاصله‌های نادرست
        if self.fix_half_space:
            text = self._fix_half_spaces(text)

        return text

    def _convert_arabic_chars(self, text):
        """
        تبدیل حروف عربی به فارسی (مثل ي → ی و ك → ک)

        :param text: متن ورودی
        :return: متن اصلاح‌شده
        """
        text = re.sub(r"[ي]", "ی", text)
        text = re.sub(r"[ك]", "ک", text)
        return text

    def _convert_numbers(self, text):
        """
        تبدیل اعداد فارسی و عربی به انگلیسی

        :param text: متن ورودی
        :return: متن با اعداد انگلیسی
        """
        arabic_to_english = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
        return text.translate(arabic_to_english)

    def _remove_extra_spaces(self, text):
        """
        حذف فاصله‌های اضافی

        :param text: متن ورودی
        :return: متن با فاصله‌های بهینه‌شده
        """
        text = re.sub(r"\s+", " ", text)  # جایگزینی فاصله‌های چندتایی با یک فاصله
        return text.strip()

    def _fix_half_spaces(self, text):
        """
        اصلاح نیم‌فاصله‌های نادرست

        :param text: متن ورودی
        :return: متن با نیم‌فاصله‌های استاندارد
        """
        text = re.sub(r"\s+‌", "‌", text)  # حذف فاصله‌های اضافی قبل از نیم‌فاصله
        text = re.sub(r"‌\s+", "‌", text)  # حذف فاصله‌های اضافی بعد از نیم‌فاصله
        return text


# مثال استفاده
if __name__ == "__main__":
    normalizer = PersianNormalizer()
    text = "این یک متن  تستی است.  يكي از مشکلات  كدهای پردازش متن، نيم فاصله  نادرست است."
    print(normalizer.normalize(text))
