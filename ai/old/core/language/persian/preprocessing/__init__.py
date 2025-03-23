# ai/core/language/persian/preprocessing/__init__.py

from .normalizer import PersianNormalizer
from .spellchecker import PersianSpellChecker
from .stopwords import PersianStopWords
from .tokenizer import PersianTokenizer

__all__ = ["PersianNormalizer", "PersianSpellChecker", "PersianStopWords", "PersianTokenizer"]

# مثال استفاده
if __name__ == "__main__":
    normalizer = PersianNormalizer()
    tokenizer = PersianTokenizer()
    stopwords = PersianStopWords()

    text = "این یک متن تستی است که باید پردازش شود."

    normalized_text = normalizer.normalize(text)
    tokens = tokenizer.tokenize(normalized_text)
    clean_text = stopwords.remove_stopwords(" ".join(tokens))

    print("متن نرمال‌شده:", normalized_text)
    print("توکن‌ها:", tokens)
    print("متن بدون کلمات توقف:", clean_text)
