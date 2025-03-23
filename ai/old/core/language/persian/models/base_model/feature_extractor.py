import torch
from models.base_model.config import BaseModelConfig
from knowledge.knowledge_store import KnowledgeStore
from knowledge.grammar import GrammarAnalyzer
from knowledge.semantic import SemanticAnalyzer
from knowledge.literature import LiteratureKnowledge
from knowledge.dialects import DialectKnowledge
from pymilvus import Collection


class FeatureExtractor:
    """
    کلاس مدیریت استخراج ویژگی‌های زبانی، معنایی و سبکی از `knowledge/`.
    """

    def __init__(self):
        """
        مقداردهی اولیه و اتصال به پایگاه داده‌های دانش.
        """
        self.knowledge_store = KnowledgeStore()
        self.grammar_analyzer = GrammarAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.literature_knowledge = LiteratureKnowledge()
        self.dialect_knowledge = DialectKnowledge()

        # اتصال به Milvus برای Embeddingها
        self.milvus_collection = Collection(name="knowledge_vectors")

    def extract_grammar_features(self, text):
        """
        استخراج ویژگی‌های دستوری از `grammar.py`.

        :param text: متن ورودی
        :return: تحلیل اشتباهات گرامری
        """
        return self.grammar_analyzer.analyze_grammar(text)

    def extract_semantic_features(self, text):
        """
        استخراج ویژگی‌های معنایی از `semantic.py`.

        :param text: متن ورودی
        :return: بردارهای معنایی مرتبط
        """
        return self.semantic_analyzer.analyze_semantics(text)

    def extract_literary_features(self, text):
        """
        استخراج ویژگی‌های سبکی و ادبی از `literature.py`.

        :param text: متن ورودی
        :return: سبک ادبی نزدیک به متن ورودی
        """
        return self.literature_knowledge.analyze_literary_style(text)

    def extract_dialect_features(self, text):
        """
        استخراج ویژگی‌های مرتبط با لهجه از `dialects.py`.

        :param text: متن ورودی
        :return: شناسایی لهجه و تبدیل به فارسی معیار
        """
        dialect = self.dialect_knowledge.get_dialect_translation("PERSIAN_STANDARD", text)
        return dialect if dialect else text

    def extract_vector_embeddings(self, text):
        """
        جستجوی Embeddingهای معنایی متن در `Milvus`.

        :param text: متن ورودی
        :return: بردارهای معنایی نزدیک به متن
        """
        query_vector = self.knowledge_store.get_text_embedding(text)
        results = self.milvus_collection.search(
            data=[query_vector],
            anns_field="vector",
            limit=5,
            output_fields=["concept"]
        )
        return [hit.entity.get("concept") for hit in results[0]]

    def extract_all_features(self, text):
        """
        استخراج کلیه ویژگی‌های مرتبط با متن.

        :param text: متن ورودی
        :return: دیکشنری شامل ویژگی‌های استخراج‌شده
        """
        return {
            "grammar": self.extract_grammar_features(text),
            "semantics": self.extract_semantic_features(text),
            "literature": self.extract_literary_features(text),
            "dialect": self.extract_dialect_features(text),
            "embedding_matches": self.extract_vector_embeddings(text)
        }


# ==================== تست ====================
if __name__ == "__main__":
    extractor = FeatureExtractor()

    test_text = "زبان فارسی یکی از قدیمی‌ترین زبان‌های جهان است."
    features = extractor.extract_all_features(test_text)

    print("📌 ویژگی‌های استخراج‌شده:", features)
