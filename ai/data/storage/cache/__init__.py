from .redis_manager import RedisManager
from .memory_cache import MemoryCache
from .distributed_cache import DistributedCache
from typing import Optional

# ایجاد نمونه‌های کش مختلف
redis_cache = RedisManager()
memory_cache = MemoryCache()
distributed_cache: Optional[DistributedCache] = None


def initialize_distributed_cache(redis_nodes: list):
    """
    مقداردهی اولیه کش توزیع‌شده با لیست نودهای Redis.

    :param redis_nodes: لیست آدرس نودهای Redis Cluster (مثلاً ["127.0.0.1:7000", "127.0.0.1:7001"])
    """
    global distributed_cache
    distributed_cache = DistributedCache(redis_nodes)


def get_cache_instance(cache_type: str = "memory"):
    """
    دریافت نمونه کش مناسب بر اساس نوع کش.

    :param cache_type: یکی از مقادیر ["memory", "redis", "distributed"]
    :return: نمونه کش موردنظر
    """
    if cache_type == "redis":
        return redis_cache
    elif cache_type == "distributed" and distributed_cache:
        return distributed_cache
    return memory_cache
