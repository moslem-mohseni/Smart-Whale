from .common import KnowledgeGraph
from .domain import DomainKnowledge
from .literature import LiteratureKnowledge
from .semantic import SemanticAnalyzer
from .contextual import ContextualKnowledge
from .grammar import GrammarAnalyzer
from .proverbs import ProverbsKnowledge
from .dialects import DialectKnowledge
from .knowledge_store import KnowledgeStore

__all__ = [
    "KnowledgeGraph",
    "DomainKnowledge",
    "LiteratureKnowledge",
    "SemanticAnalyzer",
    "ContextualKnowledge",
    "GrammarAnalyzer",
    "ProverbsKnowledge",
    "DialectKnowledge",
    "KnowledgeStore"
]

# =========================== TEST ===========================
if __name__ == "__main__":
    print("📌 تست مقداردهی اولیه‌ی `knowledge/`")

    # مقداردهی اولیه ماژول‌ها
    kg = KnowledgeGraph()
    dk = DomainKnowledge()
    lk = LiteratureKnowledge()
    sa = SemanticAnalyzer()
    ck = ContextualKnowledge()
    ga = GrammarAnalyzer()
    pk = ProverbsKnowledge()
    dkt = DialectKnowledge()
    ks = KnowledgeStore()

    # تست نمونه‌ای از عملکرد ماژول‌ها
    kg.add_node("GENERAL", "هوش مصنوعی")
    print("📌 دانش عمومی:", kg.get_nodes("GENERAL"))

    dk.add_domain_concept("MEDICINE", "پزشکی عمومی")
    print("📌 دانش پزشکی:", dk.get_domain_concepts("MEDICINE"))

    lk.add_literary_work("POETRY", "دیوان حافظ", "حافظ", "غزل")
    print("📌 آثار شعری:", lk.get_literary_works("POETRY"))

    sa.get_embedding("هوش مصنوعی چیست؟")
    print("📌 شباهت معنایی:", sa.semantic_similarity("هوش مصنوعی چیست؟", "تعریف هوش مصنوعی چیست؟"))

    ck.store_context("user_123", "کاربر درباره‌ی یادگیری ماشین سوال کرده است.", storage="redis")
    print("📌 دانش زمینه‌ای:", ck.get_context("user_123", storage="redis"))

    ga.correct_text("او رفتن به مدرسه")
    print("📌 متن اصلاح‌شده:", ga.get_correction("او رفتن به مدرسه"))

    pk.add_proverb("کار نیکو کردن از پر کردن است", "تکرار زیاد باعث مهارت می‌شود.")
    print("📌 معنی ضرب‌المثل:", pk.get_proverb_meaning("کار نیکو کردن از پر کردن است"))

    dkt.add_dialect_entry("SHIRAZI", "برادر", "دده")
    print("📌 ترجمه‌ی شیرازی:", dkt.get_dialect_translation("SHIRAZI", "برادر"))

    print("📌 مقداردهی اولیه و تست‌های `knowledge/` با موفقیت انجام شد.")
