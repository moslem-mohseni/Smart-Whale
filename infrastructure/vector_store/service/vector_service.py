from typing import List
from pymilvus import Collection
from ..adapters import ConnectionPool, RetryMechanism
from ..domain.models import Vector
from ..domain.collection import MilvusCollectionManager
from ..monitoring.performance_logger import log_insert, log_delete, log_search


class VectorService:
    """سرویس مدیریت بردارها در Milvus"""

    def __init__(self, collection_name: str = None):
        self.collection_manager = MilvusCollectionManager(collection_name)
        self.connection_pool = ConnectionPool()
        self.retry = RetryMechanism()

    @log_insert
    @RetryMechanism().retry
    def insert_vectors(self, vectors: List[Vector]):
        """افزودن بردارها به Collection در Milvus"""
        collection = Collection(self.collection_manager.collection_name)
        data = [[v.id for v in vectors], [v.values for v in vectors], [v.metadata for v in vectors]]
        collection.insert(data)
        print(f"✅ {len(vectors)} بردار با موفقیت اضافه شد.")

    @log_delete
    @RetryMechanism().retry
    def delete_vectors(self, ids: List[str]):
        """حذف بردارها از Collection بر اساس ID"""
        collection = Collection(self.collection_manager.collection_name)
        expr = f"id in {ids}"
        collection.delete(expr)
        print(f"🗑️ {len(ids)} بردار حذف شد.")

    @log_search
    @RetryMechanism().retry
    def search_vectors(self, query_vector: Vector, top_k: int = 5):
        """جستجوی نزدیک‌ترین بردارها بر اساس ANN"""
        collection = Collection(self.collection_manager.collection_name)
        collection.load()  # بارگذاری Collection در حافظه برای جستجو سریع‌تر
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_vector.values],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=["id", "metadata"]
        )
        return results
