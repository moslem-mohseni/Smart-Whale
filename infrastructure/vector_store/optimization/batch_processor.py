from typing import List
from pymilvus import Collection
from ..service.vector_service import VectorService
from .cache_manager import CacheManager
from ..adapters import RetryMechanism
from ..domain.models import Vector


class BatchProcessor:
    """پردازش دسته‌ای بردارها برای بهینه‌سازی عملکرد Milvus"""

    def __init__(self, collection_name: str = None):
        self.vector_service = VectorService(collection_name)
        self.cache_manager = CacheManager()
        self.retry = RetryMechanism()

    @RetryMechanism().retry
    async def insert_batch(self, vectors: List[Vector], batch_size: int = 100):
        """افزودن دسته‌ای بردارها به Milvus"""
        collection = Collection(self.vector_service.collection_manager.collection_name)
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            collection.insert([[v.id for v in batch], [v.values for v in batch], [v.metadata for v in batch]])
            print(f"✅ {len(batch)} بردار به صورت دسته‌ای اضافه شد.")

    @RetryMechanism().retry
    async def search_batch(self, query_vectors: List[Vector], top_k: int = 5):
        """جستجوی دسته‌ای بردارها در Milvus"""
        collection = Collection(self.vector_service.collection_manager.collection_name)
        collection.load()

        results = []
        for query_vector in query_vectors:
            cached_result = await self.cache_manager.get_cached_result(query_vector.id)
            if cached_result:
                results.append(cached_result)
                continue

            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            search_result = collection.search(
                data=[query_vector.values],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=None,
                output_fields=["id", "metadata"]
            )
            results.append(search_result)
            await self.cache_manager.cache_result(query_vector.id, search_result)

        return results
