from ..services.embedding_service import EmbeddingStore
from ..services.vector_search import VectorSearch
from ..services.knowledge_store import KnowledgeGraph
from hazm import Normalizer


class SemanticAnalyzer:
    """
    پردازش معنایی و یادگیری تدریجی مدل
    """

    def __init__(self, model_name="Smart-Persian-NLP"):
        # تنظیم مدل اولیه و سرویس‌های مورد نیاز
        self.embedding_store = EmbeddingStore()
        self.vector_search = VectorSearch()
        self.knowledge_graph = KnowledgeGraph()
        self.normalizer = Normalizer()
        self.model_name = model_name

    def get_embedding(self, text, use_parsbert=True):
        """
        دریافت بردار معنایی متن، در ابتدا از ParsBERT استفاده می‌شود اما مدل ما به مرور مستقل می‌شود
        :param text: ورودی متنی
        :param use_parsbert: اگر True باشد، از ParsBERT برای یادگیری اولیه استفاده می‌شود
        :return: بردار تعبیه‌شده متن
        """
        text = self.normalizer.normalize(text)

        # اگر مدل ما به سطح مطلوب رسیده باشد، دیگر از ParsBERT استفاده نمی‌کنیم
        if not use_parsbert:
            return self.embedding_store.get_embedding(text, model=self.model_name)

        # استفاده از ParsBERT برای تولید بردارهای اولیه
        embedding = self.embedding_store.get_embedding(text, model="ParsBERT")

        # ذخیره این داده برای یادگیری مدل اختصاصی
        self.embedding_store.save_embedding(text, embedding, model=self.model_name)
        return embedding

    def semantic_similarity(self, text1, text2):
        """
        محاسبه میزان شباهت معنایی بین دو متن
        :param text1: جمله اول
        :param text2: جمله دوم
        :return: امتیاز شباهت معنایی (0 تا 1)
        """
        vec1 = self.get_embedding(text1)
        vec2 = self.get_embedding(text2)
        return self.vector_search.cosine_similarity(vec1, vec2)

    def find_similar_texts(self, query, top_k=5):
        """
        پیدا کردن متون مشابه بر اساس جستجوی برداری
        :param query: متن ورودی
        :param top_k: تعداد متون مشابه برتر
        :return: لیستی از متون مشابه
        """
        query_vec = self.get_embedding(query)
        return self.vector_search.find_similar(query_vec, top_k=top_k)

    def update_knowledge(self, text, concept):
        """
        اضافه کردن دانش جدید به مدل ما از طریق ارتباط معنایی
        :param text: متن ورودی جدید
        :param concept: مفهوم مرتبط
        :return: وضعیت ثبت داده
        """
        return self.knowledge_graph.add_relation(text, concept)


# =========================== TEST ===========================
if __name__ == "__main__":
    analyzer = SemanticAnalyzer()

    # دریافت بردار متنی
    emb = analyzer.get_embedding("هوش مصنوعی چیست؟")
    print("📌 بردار تعبیه‌شده:", emb)

    # محاسبه شباهت معنایی
    sim_score = analyzer.semantic_similarity("هوش مصنوعی چیست؟", "تعریف هوش مصنوعی چیست؟")
    print(f"📌 میزان شباهت معنایی: {sim_score:.2f}")

    # جستجوی متون مشابه
    similar_texts = analyzer.find_similar_texts("هوش مصنوعی در چه کاربردهایی استفاده می‌شود؟")
    print("📌 متون مشابه:", similar_texts)

    # به‌روزرسانی دانش
    update_status = analyzer.update_knowledge("ماشین لرنینگ", "هوش مصنوعی")
    print("📌 وضعیت ثبت دانش:", update_status)
