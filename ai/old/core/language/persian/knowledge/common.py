from ..services.redis_service import RedisClient
from ..services.clickhouse_service import ClickHouseClient
import json


class KnowledgeGraph:
    """
    مدیریت گراف دانش و دسته‌بندی‌های معنایی.
    """

    # دسته‌بندی‌های اصلی دانش
    CATEGORIES = ["GENERAL", "LINGUISTIC", "CULTURAL", "CONCEPTUAL"]

    def __init__(self):
        self.redis = RedisClient()
        self.clickhouse = ClickHouseClient()
        self._cache = {}  # کش داخلی برای بهینه‌سازی خواندن داده‌ها

    def add_node(self, category, concept):
        """
        اضافه کردن یک مفهوم جدید به گراف دانش.

        :param category: دسته‌بندی مفهوم (GENERAL, LINGUISTIC, CULTURAL, CONCEPTUAL)
        :param concept: نام مفهوم
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"❌ دسته‌بندی نامعتبر! دسته‌های مجاز: {self.CATEGORIES}")

        # ذخیره در Redis (دسترسی سریع)
        self.redis.sadd(f"knowledge:{category}", concept)

        # ذخیره در ClickHouse (داده‌های تحلیلی)
        self.clickhouse.insert("knowledge_graph", {"category": category, "concept": concept})

        # اضافه به کش داخلی
        self._cache.setdefault(category, set()).add(concept)

    def add_relation(self, concept1, concept2, relation_type="RELATED"):
        """
        ایجاد رابطه بین دو مفهوم در گراف دانش.

        :param concept1: مفهوم اول
        :param concept2: مفهوم دوم
        :param relation_type: نوع رابطه (مثلاً "SIMILAR", "OPPOSITE", "RELATED")
        """
        relation_key = f"relation:{concept1}:{concept2}"
        relation_data = json.dumps({"concept1": concept1, "concept2": concept2, "relation": relation_type})

        # ذخیره در Redis
        self.redis.set(relation_key, relation_data)

        # ذخیره در ClickHouse
        self.clickhouse.insert("knowledge_relations",
                               {"concept1": concept1, "concept2": concept2, "relation": relation_type})

    def get_nodes(self, category):
        """
        دریافت تمام مفاهیم موجود در یک دسته.

        :param category: دسته‌بندی مورد نظر
        :return: لیست مفاهیم
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"❌ دسته‌بندی نامعتبر! دسته‌های مجاز: {self.CATEGORIES}")

        # ابتدا بررسی کش داخلی
        if category in self._cache:
            return list(self._cache[category])

        # خواندن از Redis
        concepts = self.redis.smembers(f"knowledge:{category}")
        self._cache[category] = set(concepts)
        return list(concepts)

    def get_relations(self, concept):
        """
        دریافت تمام روابط یک مفهوم.

        :param concept: مفهومی که می‌خواهیم روابط آن را دریافت کنیم.
        :return: لیستی از روابط معنایی
        """
        pattern = f"relation:{concept}:*"
        relation_keys = self.redis.keys(pattern)

        relations = []
        for key in relation_keys:
            relation_data = json.loads(self.redis.get(key))
            relations.append(relation_data)

        return relations

    def export_knowledge(self, filename="knowledge_export.json"):
        """
        خروجی گرفتن از دانش ذخیره‌شده.

        :param filename: نام فایل خروجی
        """
        knowledge = {category: self.get_nodes(category) for category in self.CATEGORIES}
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(knowledge, file, ensure_ascii=False, indent=4)

    def import_knowledge(self, filename="knowledge_export.json"):
        """
        وارد کردن داده‌های دانش از فایل.

        :param filename: نام فایل ورودی
        """
        with open(filename, "r", encoding="utf-8") as file:
            knowledge = json.load(file)

        for category, concepts in knowledge.items():
            for concept in concepts:
                self.add_node(category, concept)


# =========================== TEST ===========================
if __name__ == "__main__":
    kg = KnowledgeGraph()

    # اضافه کردن چند مفهوم
    kg.add_node("GENERAL", "هوش مصنوعی")
    kg.add_node("LINGUISTIC", "گرامر فارسی")
    kg.add_node("CULTURAL", "شاهنامه")

    # ایجاد رابطه بین مفاهیم
    kg.add_relation("هوش مصنوعی", "یادگیری ماشین", relation_type="RELATED")

    # دریافت اطلاعات
    print("📌 دانش عمومی:", kg.get_nodes("GENERAL"))
    print("📌 روابط مفهومی:", kg.get_relations("هوش مصنوعی"))

    # خروجی گرفتن و وارد کردن دانش
    kg.export_knowledge()
    kg.import_knowledge()
