import re
from hazm import Normalizer
from parsivar import FindStems


class PersianSpellChecker:
    def __init__(self, dictionary_path=None):
        """
        کلاس تصحیح املای متن فارسی

        :param dictionary_path: مسیر فایل دیکشنری سفارشی (در صورت وجود)
        """
        self.normalizer = Normalizer()
        self.stemmer = FindStems()
        self.word_dictionary = self._load_dictionary(dictionary_path) if dictionary_path else self._default_dictionary()

    def correct_text(self, text):
        """
        تصحیح املای متن فارسی

        :param text: متن فارسی
        :return: متن اصلاح‌شده
        """
        words = text.split()
        corrected_words = [self.correct_word(word) for word in words]
        return " ".join(corrected_words)

    def correct_word(self, word):
        """
        تصحیح یک کلمه در صورت موجود بودن در دیکشنری

        :param word: کلمه‌ی ورودی
        :return: کلمه‌ی تصحیح‌شده
        """
        normalized_word = self.normalizer.normalize(word)

        # اگر کلمه در لغت‌نامه باشد، بازگردانده شود
        if normalized_word in self.word_dictionary:
            return normalized_word

        # در غیر این صورت، نزدیک‌ترین کلمه را پیدا کنیم
        return self._find_closest_match(normalized_word)

    def _load_dictionary(self, dictionary_path):
        """
        بارگذاری دیکشنری سفارشی از فایل

        :param dictionary_path: مسیر فایل دیکشنری
        :return: مجموعه‌ای از کلمات معتبر
        """
        try:
            with open(dictionary_path, "r", encoding="utf-8") as file:
                return set(file.read().splitlines())
        except FileNotFoundError:
            print("⚠ دیکشنری یافت نشد، از دیکشنری پیش‌فرض استفاده می‌شود.")
            return self._default_dictionary()

    def _default_dictionary(self):
        """
        دیکشنری پیش‌فرض شامل کلمات رایج فارسی

        :return: مجموعه‌ای از کلمات معتبر
        """
        return {"کتاب", "خانه", "مدرسه", "ماشین", "دانشگاه", "سیستم", "برنامه", "نرم‌افزار", "هوش", "مصنوعی"}

    def _find_closest_match(self, word):
        """
        یافتن نزدیک‌ترین کلمه به کلمه‌ی اشتباه

        :param word: کلمه‌ی ورودی
        :return: نزدیک‌ترین کلمه‌ی ممکن
        """
        closest_word = min(self.word_dictionary, key=lambda w: self._levenshtein_distance(word, w))
        return closest_word

    def _levenshtein_distance(self, word1, word2):
        """
        محاسبه فاصله‌ی لوین‌اشتاین بین دو کلمه

        :param word1: کلمه‌ی اول
        :param word2: کلمه‌ی دوم
        :return: فاصله‌ی عددی بین دو کلمه
        """
        len1, len2 = len(word1), len(word2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            for j in range(len2 + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[len1][len2]

# مثال استفاده
if __name__ == "__main__":
    spell_checker = PersianSpellChecker()
    text = "این یک متن تسی است که نیابزم به صیح دارد."
    print(spell_checker.correct_text(text))
