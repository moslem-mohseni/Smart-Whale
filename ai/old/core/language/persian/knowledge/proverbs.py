from ..services.redis_service import RedisClient
from ..services.clickhouse_service import ClickHouseClient
from ..services.vector_search import VectorSearch
import json


class ProverbsKnowledge:
    """
    مدیریت ضرب‌المثل‌ها، تشخیص، تحلیل و پردازش معنایی آنها.
    """

    def __init__(self):
        self.redis = RedisClient()
        self.clickhouse = ClickHouseClient()
        self.vector_search = VectorSearch()
        self._cache = {}  # کش داخلی برای کاهش درخواست‌های پایگاه داده

    def add_proverb(self, proverb, meaning):
        """
        اضافه کردن یک ضرب‌المثل جدید به پایگاه دانش.

        :param proverb: متن ضرب‌المثل
        :param meaning: معنی ساده‌ی ضرب‌المثل
        """
        # ذخیره در Redis برای جستجوی سریع
        self.redis.set(f"proverb:{proverb}", meaning)

        # ذخیره در ClickHouse برای تحلیل داده‌ها
        self.clickhouse.insert("proverbs_knowledge", {"proverb": proverb, "meaning": meaning})

        # اضافه به کش داخلی
        self._cache[proverb] = meaning

    def get_proverb_meaning(self, proverb):
        """
        دریافت معنی یک ضرب‌المثل.

        :param proverb: متن ضرب‌المثل
        :return: معنی ساده‌ی آن
        """
        # بررسی کش داخلی
        if proverb in self._cache:
            return self._cache[proverb]

        # دریافت از Redis
        meaning = self.redis.get(f"proverb:{proverb}")
        if meaning:
            self._cache[proverb] = meaning
            return meaning

        return "❌ معنی این ضرب‌المثل در پایگاه دانش موجود نیست."

    def find_similar_proverbs(self, query, top_k=5):
        """
        جستجوی ضرب‌المثل‌های مشابه بر اساس پردازش برداری.

        :param query: متن ورودی
        :param top_k: تعداد ضرب‌المثل‌های مشابه
        :return: لیستی از ضرب‌المثل‌های مرتبط
        """
        query_vec = self.vector_search.get_embedding(query)
        return self.vector_search.find_similar(query_vec, top_k=top_k)

    def export_proverbs(self, filename="proverbs.json"):
        """
        خروجی گرفتن از پایگاه داده‌ی ضرب‌المثل‌ها.

        :param filename: نام فایل خروجی
        """
        proverbs = self.redis.keys("proverb:*")
        data = {p.replace("proverb:", ""): self.redis.get(p) for p in proverbs}

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def import_proverbs(self, filename="proverbs.json"):
        """
        وارد کردن داده‌های ضرب‌المثل از فایل JSON.

        :param filename: نام فایل ورودی
        """
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        for proverb, meaning in data.items():
            self.add_proverb(proverb, meaning)


# =========================== TEST ===========================
if __name__ == "__main__":
    pk = ProverbsKnowledge()

    # اضافه کردن ضرب‌المثل‌های جدید
    pk.add_proverb("کار نیکو کردن از پر کردن است", "تکرار زیاد باعث مهارت می‌شود.")
    pk.add_proverb("آب که از سر گذشت، چه یک وجب چه صد وجب", "وقتی کاری از کنترل خارج شد، شدت آن دیگر مهم نیست.")

    # دریافت معنی ضرب‌المثل
    print("📌 معنی ضرب‌المثل:", pk.get_proverb_meaning("کار نیکو کردن از پر کردن است"))

    # جستجوی ضرب‌المثل‌های مشابه
    print("📌 ضرب‌المثل‌های مشابه:", pk.find_similar_proverbs("تمرین زیاد باعث پیشرفت می‌شود."))

    # خروجی گرفتن و وارد کردن داده‌ها
    pk.export_proverbs()
    pk.import_proverbs()
