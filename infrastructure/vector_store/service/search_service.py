from typing import List, Dict
from pymilvus import Collection
from ..adapters import ConnectionPool, RetryMechanism
from ..domain.models import Vector
from ..domain.collection import MilvusCollectionManager


class SearchService:
    """سرویس جستجوی بردارها در Milvus"""

    def __init__(self, collection_name: str = None):
        self.collection_manager = MilvusCollectionManager(collection_name)
        self.connection_pool = ConnectionPool()
        self.retry = RetryMechanism()

    @RetryMechanism().retry
    def similarity_search(self, query_vector: Vector, top_k: int = 5):
        """جستجوی شباهتی برای یافتن نزدیک‌ترین بردارها"""
        collection = Collection(self.collection_manager.collection_name)
        collection.load()
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

    @RetryMechanism().retry
    def range_search(self, query_vector: Vector, radius: float):
        """جستجوی بردارها در یک فاصله مشخص از بردار ورودی"""
        collection = Collection(self.collection_manager.collection_name)
        collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = collection.search(
            data=[query_vector.values],
            anns_field="vector",
            param=search_params,
            limit=100,  # حداکثر مقدار اولیه
            expr=f"distance(vector, {query_vector.values}) <= {radius}",
            output_fields=["id", "metadata"]
        )
        return results

    @RetryMechanism().retry
    def hybrid_search(self, query_vector: Vector, filters: Dict, top_k: int = 5):
        """جستجوی ترکیبی شامل فیلترهای متنی و عددی همراه با ANN"""
        collection = Collection(self.collection_manager.collection_name)
        collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        filter_expr = " AND ".join([f"{key} == {value}" for key, value in filters.items()])

        results = collection.search(
            data=[query_vector.values],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["id", "metadata"]
        )
        return results
