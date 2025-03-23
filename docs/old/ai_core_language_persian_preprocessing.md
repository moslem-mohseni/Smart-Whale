# ماژول پیش‌پردازش زبان فارسی (Preprocessing)

## هدف ماژول

ماژول `preprocessing` در مسیر زیر قرار دارد:

📂 **مسیر ماژول:** `ai/core/language/persian/preprocessing/`

این ماژول مسئول **پردازش‌های اولیه روی متون فارسی** است که شامل **نرمال‌سازی، تصحیح املایی، حذف کلمات توقف و توکن‌سازی** می‌شود. این پردازش‌ها به بهبود کیفیت داده‌های ورودی به مدل‌های پردازش زبان طبیعی (NLP) کمک می‌کنند.

---

## **ساختار فایل‌ها**

```
preprocessing/
    │── __init__.py           # مقداردهی اولیه ماژول
    │── normalizer.py        # نرمال‌سازی متون فارسی
    │── spellchecker.py      # اصلاح غلط‌های املایی
    │── stopwords.py         # حذف توقف‌واژه‌ها
    │── tokenizer.py         # توکن‌سازی متون فارسی
```

---

## **شرح فایل‌ها و عملکرد آن‌ها**

### **1️⃣ `normalizer.py` - ماژول نرمال‌سازی متن**

📌 **هدف:** استانداردسازی متون فارسی با حذف نویزها و اصلاح مشکلات نگارشی.

**ویژگی‌ها:**

- تبدیل اعداد عربی به فارسی
- حذف فاصله‌های اضافی و نیم‌فاصله‌های نادرست
- اصلاح حروف عربی (مثلاً "ي" به "ی" و "ك" به "ک")
- نرمال‌سازی کلی متن

**متدهای کلیدی:**

- `normalize(text)`: اعمال تمامی مراحل نرمال‌سازی روی متن ورودی
- `_convert_arabic_chars(text)`: تبدیل حروف عربی به فارسی
- `_convert_numbers(text)`: تبدیل اعداد فارسی و عربی به انگلیسی
- `_remove_extra_spaces(text)`: حذف فاصله‌های اضافی
- `_fix_half_spaces(text)`: اصلاح نیم‌فاصله‌های نادرست

---

### **2️⃣ `spellchecker.py` - ماژول اصلاح غلط‌های املایی**

📌 **هدف:** اصلاح غلط‌های متداول املایی در متون فارسی با استفاده از الگوریتم‌های تصحیح املایی.

**ویژگی‌ها:**

- تشخیص و اصلاح کلمات دارای غلط املایی
- پیشنهاد نزدیک‌ترین کلمه‌ی صحیح از دیکشنری
- امکان بارگذاری دیکشنری سفارشی

**متدهای کلیدی:**

- `correct_text(text)`: اصلاح غلط‌های املایی در متن ورودی
- `correct_word(word)`: تصحیح یک کلمه بر اساس نزدیک‌ترین تطبیق در دیکشنری
- `_load_dictionary(dictionary_path)`: بارگذاری دیکشنری سفارشی از فایل
- `_default_dictionary()`: استفاده از دیکشنری پیش‌فرض
- `_find_closest_match(word)`: پیدا کردن نزدیک‌ترین کلمه برای تصحیح
- `_levenshtein_distance(word1, word2)`: محاسبه فاصله‌ی لوین‌اشتاین بین دو کلمه

---

### **3️⃣ `stopwords.py` - ماژول حذف توقف‌واژه‌ها**

📌 **هدف:** حذف کلمات پرتکرار و کم‌اهمیت در زبان فارسی که تأثیر معنایی زیادی ندارند.

**ویژگی‌ها:**

- حذف **توقف‌واژه‌های فارسی** مانند "از", "به", "که", "در", "و", "را" و ...
- امکان افزودن یا حذف کلمات سفارشی از لیست توقف‌واژه‌ها

**متدهای کلیدی:**

- `remove_stopwords(text)`: حذف توقف‌واژه‌ها از متن ورودی
- `_load_stopwords(stopwords_path)`: بارگذاری لیست توقف‌واژه‌ها از فایل
- `_default_stopwords()`: استفاده از لیست پیش‌فرض توقف‌واژه‌ها

---

### **4️⃣ `tokenizer.py` - ماژول توکن‌سازی**

📌 **هدف:** تبدیل متن به **واحدهای کوچک‌تر (توکن‌ها)** برای پردازش زبان طبیعی.

**ویژگی‌ها:**

- **تقسیم متن به کلمات** (Word Tokenization)
- **تقسیم متن به جملات** (Sentence Tokenization)
- امکان پردازش جداگانه علائم نگارشی

**متدهای کلیدی:**

- `tokenize(text)`: تبدیل متن ورودی به لیست توکن‌ها
- `_normalize(text)`: نرمال‌سازی اولیه متن قبل از توکن‌سازی
- `_split_punctuation(tokens)`: جداسازی علائم نگارشی از کلمات

---

## **مقداردهی اولیه ماژول**

📂 **`__init__.py`**

```python
from .normalizer import PersianNormalizer
from .spellchecker import PersianSpellChecker
from .stopwords import PersianStopWords
from .tokenizer import PersianTokenizer

__all__ = ["PersianNormalizer", "PersianSpellChecker", "PersianStopWords", "PersianTokenizer"]
```

این فایل **ماژول‌های اصلی را مقداردهی** می‌کند و دسترسی مستقیم به کلاس‌های مهم را از طریق `preprocessing` ممکن می‌سازد.

---

## **مثال استفاده از ماژول**

```python
from ai.core.language.persian.preprocessing import PersianNormalizer, PersianSpellChecker, PersianStopWords, PersianTokenizer

text = "این یک متن تستی است که باید پردازش شود."

# ایجاد نمونه از کلاس‌های مورد نیاز
normalizer = PersianNormalizer()
tokenizer = PersianTokenizer()
stopwords = PersianStopWords()
spellchecker = PersianSpellChecker()

# نرمال‌سازی متن
normalized_text = normalizer.normalize(text)

# توکن‌سازی
tokens = tokenizer.tokenize(normalized_text)

# حذف کلمات توقف
clean_text = stopwords.remove_stopwords(" ".join(tokens))

# تصحیح غلط‌های املایی
final_text = spellchecker.correct_text(clean_text)

print("متن نرمال‌شده:", normalized_text)
print("توکن‌ها:", tokens)
print("متن بدون کلمات توقف:", clean_text)
print("متن نهایی پس از تصحیح املایی:", final_text)
```

---

## **نقشه راه توسعه**

✅ **بهینه‌سازی فرآیندهای پردازش متون**
✅ **افزودن مدل یادگیری عمیق برای اصلاح غلط‌های املایی**
✅ **بهبود عملکرد توکنایزر برای پردازش متون تخصصی**
✅ **افزودن قابلیت تحلیل و پردازش متون با علائم خاص (شعر، مکالمات، ایمیل‌ها)**

