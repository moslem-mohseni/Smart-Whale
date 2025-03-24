from typing import List, Dict, Any
from pymilvus import Collection
from .cache_manager import CacheManager
from ..adapters import RetryMechanism
from ..domain.models import Vector


class QueryOptimizer:
    """بهینه‌سازی کوئری‌های جستجو در Milvus"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name
        self.cache_manager = CacheManager()
        self.retry = RetryMechanism()

    def optimize_search_params(self, data_size: int) -> Dict[str, Any]:
        """
        تعیین مقادیر بهینه برای `nprobe` و `efSearch`
        :param data_size: تعداد داده‌های ذخیره‌شده
        :return: تنظیمات بهینه برای جستجو
        """
        if data_size < 100_000:
            return {"metric_type": "L2", "params": {"nprobe": 10, "efSearch": 50}}
        elif data_size < 1_000_000:
            return {"metric_type": "L2", "params": {"nprobe": 20, "efSearch": 100}}
        else:
            return {"metric_type": "L2", "params": {"nprobe": 40, "efSearch": 200}}

    @RetryMechanism().retry
    async def optimized_search(self, query_vectors: List[Vector], top_k: int = 5):
        """اجرای جستجو با تنظیمات بهینه‌شده"""
        collection = Collection(self.collection_name)
        collection.load()

        total_entities = collection.num_entities  # تعداد کل داده‌ها
        search_params = self.optimize_search_params(total_entities)

        results = []
        for query_vector in query_vectors:
            cached_result = await self.cache_manager.get_cached_result(query_vector.id)
            if cached_result:
                results.append(cached_result)
                continue

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
