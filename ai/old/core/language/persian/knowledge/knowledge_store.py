from ..services.redis_service import RedisClient
from ..services.clickhouse_service import ClickHouseClient
from ..services.kafka_service import KafkaProducer, KafkaConsumer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import json
import numpy as np


class KnowledgeStore:
    """
    مدیریت ذخیره‌سازی و بازیابی داده‌های دانش از پایگاه‌داده‌ها (Redis, ClickHouse, Kafka, Milvus)
    """

    def __init__(self):
        # اتصال به سرویس‌های داده
        self.redis = RedisClient()
        self.clickhouse = ClickHouseClient()
        self.kafka_producer = KafkaProducer(topic="knowledge_updates")
        self.kafka_consumer = KafkaConsumer(topic="knowledge_updates")

        # اتصال به Milvus
        connections.connect(host="localhost", port="19530")
        self.milvus_collection = self._setup_milvus_collection()

    # ==============================
    # 📌 تنظیم و مدیریت پایگاه داده Milvus
    # ==============================
    def _setup_milvus_collection(self):
        """
        تنظیم پایگاه داده‌ی Milvus برای ذخیره و جستجوی بردارها.
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="concept", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        ]
        schema = CollectionSchema(fields, description="Knowledge Embeddings")
        collection = Collection(name="knowledge_vectors", schema=schema)

        # ایجاد ایندکس برای بهبود سرعت جستجو
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        collection.create_index(field_name="vector", index_params=index_params)
        return collection

    def save_vector_embedding(self, concept, vector):
        """
        ذخیره‌ی بردار معنایی یک مفهوم در Milvus.

        :param concept: مفهوم موردنظر
        :param vector: بردار معنایی
        """
        data = [[None], [concept], [vector]]
        self.milvus_collection.insert(data)
        self.milvus_collection.flush()

    def find_similar_concepts(self, query_vector, top_k=5):
        """
        جستجوی مفاهیم مشابه از طریق Milvus.

        :param query_vector: بردار متنی برای جستجو
        :param top_k: تعداد مفاهیم مشابه برتر
        :return: لیستی از مفاهیم مشابه
        """
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.milvus_collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["concept"],
        )
        return [hit.entity.get("concept") for hit in results[0]]

    # ==============================
    # 📌 مدیریت گراف دانش (Knowledge Graph)
    # ==============================
    def save_graph_relation(self, concept1, concept2, relation_type="RELATED"):
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

    def get_graph_relations(self, concept):
        """
        دریافت تمام روابط یک مفهوم در گراف دانش.

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

    # ==============================
    # 📌 مدیریت دانش در `Redis` و `ClickHouse`
    # ==============================
    def save_knowledge(self, key, data, storage="redis"):
        """
        ذخیره داده‌های دانش در یکی از پایگاه‌های داده.

        :param key: کلید یکتا برای دانش
        :param data: داده‌ای که ذخیره می‌شود
        :param storage: نوع پایگاه داده ("redis", "clickhouse", "kafka")
        """
        if storage == "redis":
            self.redis.set(key, data)
        elif storage == "clickhouse":
            self.clickhouse.insert("knowledge_table", {"key": key, "data": data})
        elif storage == "kafka":
            self.kafka_producer.send({"key": key, "data": data})
        else:
            raise ValueError("❌ نوع پایگاه‌داده نامعتبر است!")

    def get_knowledge(self, key, storage="redis"):
        """
        دریافت داده‌های ذخیره‌شده از پایگاه‌های داده.

        :param key: کلید دانش موردنظر
        :param storage: نوع پایگاه داده ("redis", "clickhouse")
        :return: داده‌ی ذخیره‌شده
        """
        if storage == "redis":
            return self.redis.get(key)
        elif storage == "clickhouse":
            return self.clickhouse.query(f"SELECT data FROM knowledge_table WHERE key='{key}'")
        else:
            raise ValueError("❌ نوع پایگاه‌داده نامعتبر است!")

    # ==============================
    # 📌 استریم به‌روزرسانی‌های دانش از Kafka
    # ==============================
    def stream_knowledge_updates(self):
        """
        دریافت تغییرات و به‌روزرسانی‌های دانش از Kafka.
        """
        for message in self.kafka_consumer.listen():
            print(f"📌 دانش جدید دریافت شد: {message}")


# =========================== TEST ===========================
if __name__ == "__main__":
    store = KnowledgeStore()

    # ذخیره و دریافت دانش در Redis
    store.save_knowledge("context:hello", "سلام!", storage="redis")
    print("📌 دانش ذخیره‌شده در Redis:", store.get_knowledge("context:hello", storage="redis"))

    # ایجاد رابطه در گراف دانش
    store.save_graph_relation("هوش مصنوعی", "یادگیری ماشین", relation_type="RELATED")
    print("📌 روابط هوش مصنوعی:", store.get_graph_relations("هوش مصنوعی"))

    # ذخیره و جستجوی بردارهای معنایی با Milvus
    sample_vector = np.random.rand(768).tolist()
    store.save_vector_embedding("یادگیری ماشین", sample_vector)
    print("📌 مفاهیم مشابه:", store.find_similar_concepts(sample_vector))

    # استریم به‌روزرسانی‌های دانش
    store.stream_knowledge_updates()
