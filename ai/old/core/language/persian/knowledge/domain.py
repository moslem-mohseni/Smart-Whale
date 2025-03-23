from ..services.redis_service import RedisClient
from ..services.clickhouse_service import ClickHouseClient
import json


class DomainKnowledge:
    """
    مدیریت دانش تخصصی در حوزه‌های مختلف.
    """

    # دسته‌بندی‌های اصلی حوزه‌های تخصصی
    DOMAINS = ["MEDICINE", "ENGINEERING", "LAW", "FINANCE", "LINGUISTICS"]

    def __init__(self):
        self.redis = RedisClient()
        self.clickhouse = ClickHouseClient()
        self._cache = {}  # کش داخلی برای جلوگیری از پردازش‌های اضافی

    def add_domain_concept(self, domain, concept, parent=None):
        """
        اضافه کردن یک مفهوم جدید به دانش تخصصی.

        :param domain: نام حوزه‌ی تخصصی (MEDICINE, ENGINEERING, ...)
        :param concept: مفهوم جدید
        :param parent: مفهوم والد (در صورت وجود)
        """
        if domain not in self.DOMAINS:
            raise ValueError(f"❌ حوزه‌ی تخصصی نامعتبر! حوزه‌های مجاز: {self.DOMAINS}")

        # ذخیره در Redis (برای دسترسی سریع)
        self.redis.sadd(f"domain:{domain}", concept)

        # ذخیره در ClickHouse (تحلیل داده‌های تخصصی)
        data = {"domain": domain, "concept": concept, "parent": parent or "ROOT"}
        self.clickhouse.insert("domain_knowledge", data)

        # اضافه به کش داخلی
        self._cache.setdefault(domain, set()).add(concept)

    def get_domain_concepts(self, domain):
        """
        دریافت تمامی مفاهیم مرتبط با یک حوزه‌ی تخصصی.

        :param domain: نام حوزه‌ی تخصصی
        :return: لیست مفاهیم مرتبط
        """
        if domain not in self.DOMAINS:
            raise ValueError(f"❌ حوزه‌ی تخصصی نامعتبر! حوزه‌های مجاز: {self.DOMAINS}")

        # ابتدا بررسی کش داخلی
        if domain in self._cache:
            return list(self._cache[domain])

        # خواندن از Redis
        concepts = self.redis.smembers(f"domain:{domain}")
        self._cache[domain] = set(concepts)
        return list(concepts)

    def get_domain_hierarchy(self, domain):
        """
        دریافت سلسله‌مراتب دانش در یک حوزه‌ی تخصصی.

        :param domain: نام حوزه‌ی تخصصی
        :return: ساختار سلسله‌مراتب
        """
        query = f"SELECT concept, parent FROM domain_knowledge WHERE domain='{domain}'"
        results = self.clickhouse.query(query)

        hierarchy = {}
        for row in results:
            concept, parent = row["concept"], row["parent"]
            if parent == "ROOT":
                hierarchy[concept] = []
            else:
                hierarchy.setdefault(parent, []).append(concept)

        return hierarchy

    def export_domain_knowledge(self, filename="domain_knowledge.json"):
        """
        خروجی گرفتن از دانش تخصصی ذخیره‌شده.

        :param filename: نام فایل خروجی
        """
        knowledge = {domain: self.get_domain_concepts(domain) for domain in self.DOMAINS}
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(knowledge, file, ensure_ascii=False, indent=4)

    def import_domain_knowledge(self, filename="domain_knowledge.json"):
        """
        وارد کردن داده‌های دانش تخصصی از فایل.

        :param filename: نام فایل ورودی
        """
        with open(filename, "r", encoding="utf-8") as file:
            knowledge = json.load(file)

        for domain, concepts in knowledge.items():
            for concept in concepts:
                self.add_domain_concept(domain, concept)


# =========================== TEST ===========================
if __name__ == "__main__":
    dk = DomainKnowledge()

    # اضافه کردن دانش تخصصی
    dk.add_domain_concept("MEDICINE", "پزشکی عمومی")
    dk.add_domain_concept("MEDICINE", "جراحی مغز و اعصاب", parent="پزشکی عمومی")
    dk.add_domain_concept("ENGINEERING", "مهندسی نرم‌افزار")

    # دریافت داده‌ها
    print("📌 دانش پزشکی:", dk.get_domain_concepts("MEDICINE"))
    print("📌 سلسله‌مراتب دانش پزشکی:", dk.get_domain_hierarchy("MEDICINE"))

    # خروجی گرفتن و وارد کردن دانش
    dk.export_domain_knowledge()
    dk.import_domain_knowledge()
