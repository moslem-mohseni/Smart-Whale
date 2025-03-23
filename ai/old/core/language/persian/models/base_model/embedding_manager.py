import numpy as np
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType
from models.base_model.config import BaseModelConfig
from models.base_model.feature_extractor import FeatureExtractor


class EmbeddingManager:
    """
    کلاس مدیریت ذخیره‌سازی و بازیابی Embeddingهای معنایی از `Milvus`.
    """

    def __init__(self):
        """
        مقداردهی اولیه و اتصال به پایگاه داده Milvus.
        """
        self.extractor = FeatureExtractor()

        # اتصال به Milvus
        connections.connect(host="localhost", port="19530")
        self.collection = self._setup_collection()

    def _setup_collection(self):
        """
        تنظیم پایگاه داده‌ی Milvus برای ذخیره و جستجوی بردارهای معنایی.
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        ]
        schema = CollectionSchema(fields, description="Persian NLP Embeddings")
        collection = Collection(name="persian_embeddings", schema=schema)

        # ایجاد ایندکس برای جستجوی سریع
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        collection.create_index(field_name="vector", index_params=index_params)

        return collection

    def store_embedding(self, text):
        """
        ذخیره‌ی Embedding معنایی یک متن در Milvus.

        :param text: متن ورودی برای پردازش
        """
        vector = self.extractor.extract_semantic_features(text)
        data = [[None], [text], [vector]]

        self.collection.insert(data)
        self.collection.flush()
        print(f"✅ Embedding برای '{text}' ذخیره شد.")

    def find_similar_texts(self, query_text, top_k=5):
        """
        جستجوی متون مشابه از طریق Milvus.

        :param query_text: متن ورودی
        :param top_k: تعداد نزدیک‌ترین Embeddingها
        :return: لیستی از متون مرتبط
        """
        query_vector = self.extractor.extract_semantic_features(query_text)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        return [hit.entity.get("text") for hit in results[0]]

    def load_all_embeddings(self):
        """
        دریافت تمام Embeddingهای ذخیره‌شده.
        """
        return self.collection.query(expr="", output_fields=["text", "vector"])


# ==================== تست ====================
if __name__ == "__main__":
    manager = EmbeddingManager()

    test_text = "زبان فارسی یکی از مهم‌ترین زبان‌های جهان است."
    manager.store_embedding(test_text)

    similar_texts = manager.find_similar_texts("فارسی زبان تاریخی و مهم است.")
    print("📌 متون مشابه:", similar_texts)

    all_embeddings = manager.load_all_embeddings()
    print(f"📌 تعداد Embeddingهای ذخیره‌شده: {len(all_embeddings)}")
