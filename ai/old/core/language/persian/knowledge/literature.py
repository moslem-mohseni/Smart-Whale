from ..services.redis_service import RedisClient
from ..services.clickhouse_service import ClickHouseClient
import json


class LiteratureKnowledge:
    """
    مدیریت دانش ادبیات فارسی، سبک‌های شعری و نویسندگان.
    """

    # دسته‌بندی‌های ادبی (برای تحلیل سبک‌شناسی)
    CATEGORIES = ["POETRY", "PROSE", "CLASSIC", "MODERN"]

    def __init__(self):
        self.redis = RedisClient()
        self.clickhouse = ClickHouseClient()
        self._cache = {}  # کش داخلی برای بهینه‌سازی پردازش‌ها

    def add_literary_work(self, category, title, author, style=None):
        """
        اضافه کردن یک اثر ادبی جدید.

        :param category: دسته‌بندی اثر (POETRY, PROSE, CLASSIC, MODERN)
        :param title: نام اثر
        :param author: نویسنده یا شاعر
        :param style: سبک ادبی (مثلاً "غزل" برای شعر)
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"❌ دسته‌بندی نامعتبر! دسته‌های مجاز: {self.CATEGORIES}")

        # ذخیره در Redis (برای جستجوی سریع)
        key = f"literature:{category}:{title}"
        data = {"title": title, "author": author, "style": style or "UNKNOWN"}
        self.redis.set(key, json.dumps(data))

        # ذخیره در ClickHouse (تحلیل داده‌های ادبی)
        self.clickhouse.insert("literature_knowledge",
                               {"category": category, "title": title, "author": author, "style": style or "UNKNOWN"})

        # اضافه به کش داخلی
        self._cache.setdefault(category, {}).update({title: data})

    def get_literary_works(self, category):
        """
        دریافت آثار ادبی ذخیره‌شده در یک دسته.

        :param category: دسته‌بندی اثر (POETRY, PROSE, CLASSIC, MODERN)
        :return: لیست آثار ادبی
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"❌ دسته‌بندی نامعتبر! دسته‌های مجاز: {self.CATEGORIES}")

        # بررسی کش داخلی
        if category in self._cache:
            return list(self._cache[category].values())

        # دریافت از Redis
        pattern = f"literature:{category}:*"
        works = []
        for key in self.redis.keys(pattern):
            data = json.loads(self.redis.get(key))
            works.append(data)

        # ذخیره در کش برای دسترسی سریع‌تر
        self._cache[category] = {work["title"]: work for work in works}
        return works

    def analyze_literary_style(self, text):
        """
        تحلیل سبک‌شناسی یک متن و مقایسه با داده‌های ادبی موجود.

        :param text: متن ورودی
        :return: نزدیک‌ترین سبک ادبی
        """
        # دریافت داده‌های سبک‌شناسی از ClickHouse
        query = "SELECT DISTINCT style FROM literature_knowledge"
        styles = self.clickhouse.query(query)
        styles = [row["style"] for row in styles if row["style"] != "UNKNOWN"]

        # الگوریتم ساده برای پیدا کردن سبک متنی
        best_match = "UNKNOWN"
        max_match_count = 0
        for style in styles:
            match_count = text.count(style)  # بررسی فراوانی کلمات مربوط به سبک در متن
            if match_count > max_match_count:
                best_match = style
                max_match_count = match_count

        return best_match

    def export_literature_knowledge(self, filename="literature_knowledge.json"):
        """
        خروجی گرفتن از دانش ادبی ذخیره‌شده.

        :param filename: نام فایل خروجی
        """
        knowledge = {category: self.get_literary_works(category) for category in self.CATEGORIES}
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(knowledge, file, ensure_ascii=False, indent=4)

    def import_literature_knowledge(self, filename="literature_knowledge.json"):
        """
        وارد کردن داده‌های دانش ادبی از فایل.

        :param filename: نام فایل ورودی
        """
        with open(filename, "r", encoding="utf-8") as file:
            knowledge = json.load(file)

        for category, works in knowledge.items():
            for work in works:
                self.add_literary_work(category, work["title"], work["author"], work.get("style"))


# =========================== TEST ===========================
if __name__ == "__main__":
    lk = LiteratureKnowledge()

    # اضافه کردن آثار ادبی
    lk.add_literary_work("POETRY", "دیوان حافظ", "حافظ", "غزل")
    lk.add_literary_work("PROSE", "گلستان سعدی", "سعدی", "نثر مسجع")
    lk.add_literary_work("CLASSIC", "شاهنامه", "فردوسی", "حماسی")

    # دریافت داده‌ها
    print("📌 آثار شعری:", lk.get_literary_works("POETRY"))
    print("📌 تحلیل سبک‌شناسی:", lk.analyze_literary_style("این غزل زیبا سروده‌ی حافظ است"))

    # خروجی گرفتن و وارد کردن دانش
    lk.export_literature_knowledge()
    lk.import_literature_knowledge()
