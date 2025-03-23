from ..services.redis_service import RedisClient
from ..services.clickhouse_service import ClickHouseClient
from ..services.vector_search import VectorSearch
import json


class DialectKnowledge:
    """
    مدیریت یادگیری لهجه‌های فارسی، شناسایی، دسته‌بندی و تحلیل تفاوت‌های زبانی.
    """

    DIALECTS = ["LORI", "ESFAHANI", "SHIRAZI", "SOUTHERN", "PERSIAN_STANDARD"]

    def __init__(self):
        self.redis = RedisClient()
        self.clickhouse = ClickHouseClient()
        self.vector_search = VectorSearch()
        self._cache = {}  # کش داخلی برای بهینه‌سازی پردازش‌ها

    # ==============================
    # 📌 مدیریت پایگاه داده‌ی لهجه‌ها
    # ==============================
    def add_dialect_entry(self, dialect, standard_word, dialect_word, example_sentence=None):
        """
        اضافه کردن یک واژه یا عبارت به پایگاه داده‌ی لهجه‌ها.

        :param dialect: نام لهجه (LORI, ESFAHANI, ...)
        :param standard_word: معادل فارسی معیار
        :param dialect_word: واژه در لهجه‌ی مورد نظر
        :param example_sentence: یک جمله نمونه که این واژه در آن استفاده شده است
        """
        if dialect not in self.DIALECTS:
            raise ValueError(f"❌ لهجه‌ی نامعتبر! لهجه‌های مجاز: {self.DIALECTS}")

        key = f"dialect:{dialect}:{standard_word}"
        data = {"standard": standard_word, "dialect": dialect_word, "example": example_sentence or ""}

        # ذخیره در Redis
        self.redis.set(key, json.dumps(data))

        # ذخیره در ClickHouse برای تحلیل‌های عمیق‌تر
        self.clickhouse.insert("dialect_knowledge",
                               {"dialect": dialect, "standard": standard_word, "dialect_word": dialect_word,
                                "example": example_sentence or ""})

        # اضافه به کش داخلی
        self._cache.setdefault(dialect, {})[standard_word] = data

    def get_dialect_translation(self, dialect, standard_word):
        """
        دریافت ترجمه‌ی یک واژه از فارسی معیار به لهجه‌ی مورد نظر.

        :param dialect: نام لهجه (LORI, ESFAHANI, ...)
        :param standard_word: واژه‌ی فارسی معیار
        :return: معادل آن در لهجه‌ی مورد نظر
        """
        key = f"dialect:{dialect}:{standard_word}"
        if key in self._cache.get(dialect, {}):
            return self._cache[dialect][standard_word]

        return json.loads(self.redis.get(key) or "{}")

    # ==============================
    # 📌 تحلیل تفاوت‌های لهجه‌ای
    # ==============================
    def analyze_dialect_difference(self, dialect1, dialect2):
        """
        مقایسه‌ی دو لهجه و نمایش تفاوت‌های کلیدی آنها.

        :param dialect1: لهجه‌ی اول
        :param dialect2: لهجه‌ی دوم
        :return: لیستی از واژه‌های متفاوت بین این دو لهجه
        """
        keys1 = self.redis.keys(f"dialect:{dialect1}:*")
        keys2 = self.redis.keys(f"dialect:{dialect2}:*")

        words1 = {key.split(":")[-1] for key in keys1}
        words2 = {key.split(":")[-1] for key in keys2}

        diff1 = words1 - words2
        diff2 = words2 - words1

        return {"unique_to_" + dialect1: list(diff1), "unique_to_" + dialect2: list(diff2)}

    def find_similar_dialect_words(self, query, dialect, top_k=5):
        """
        جستجوی واژه‌های مشابه در یک لهجه.

        :param query: واژه‌ی ورودی
        :param dialect: نام لهجه
        :param top_k: تعداد نتایج برتر
        :return: لیستی از واژه‌های مشابه
        """
        query_vec = self.vector_search.get_embedding(query)
        return self.vector_search.find_similar(query_vec, top_k=top_k)

    # ==============================
    # 📌 ذخیره‌سازی و بازیابی داده‌ها
    # ==============================
    def export_dialect_data(self, filename="dialects.json"):
        """
        خروجی گرفتن از پایگاه داده‌ی لهجه‌ها.

        :param filename: نام فایل خروجی
        """
        dialect_data = {}
        for dialect in self.DIALECTS:
            dialect_data[dialect] = {key.replace(f"dialect:{dialect}:", ""): json.loads(self.redis.get(key))
                                     for key in self.redis.keys(f"dialect:{dialect}:*")}

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(dialect_data, file, ensure_ascii=False, indent=4)

    def import_dialect_data(self, filename="dialects.json"):
        """
        وارد کردن داده‌های لهجه از فایل JSON.

        :param filename: نام فایل ورودی
        """
        with open(filename, "r", encoding="utf-8") as file:
            dialect_data = json.load(file)

        for dialect, words in dialect_data.items():
            for standard_word, data in words.items():
                self.add_dialect_entry(dialect, standard_word, data["dialect"], data.get("example"))


# =========================== TEST ===========================
if __name__ == "__main__":
    dp = DialectKnowledge()

    # اضافه کردن واژه‌های لهجه‌ای
    dp.add_dialect_entry("LORI", "سلام", "سلاو", "سلاو کاکا!")
    dp.add_dialect_entry("ESFAHANI", "چطوری؟", "چطورُم؟", "چطورُم داداش؟")

    # دریافت ترجمه‌ی لهجه‌ای
    print("📌 ترجمه‌ی اصفهانی:", dp.get_dialect_translation("ESFAHANI", "چطوری؟"))

    # تحلیل تفاوت‌های لهجه‌ای
    print("📌 تفاوت لهجه‌ی لری و اصفهانی:", dp.analyze_dialect_difference("LORI", "ESFAHANI"))

    # خروجی گرفتن و وارد کردن داده‌ها
    dp.export_dialect_data()
    dp.import_dialect_data()
