import gc
import time
from typing import Dict, Any
from infrastructure.redis.cache_manager import RedisCache
from infrastructure.timescaledb.timescale_manager import TimescaleDB
from infrastructure.vector_store.vector_manager import VectorStore

class MemoryOptimizer:
    """
    ماژول بهینه‌سازی حافظه که از تکنیک‌های مختلف برای بهینه‌سازی مصرف حافظه استفاده می‌کند.
    """

    def __init__(self):
        """
        مقداردهی اولیه با سرویس‌های زیرساختی.
        """
        self.redis_cache = RedisCache()
        self.timescale_db = TimescaleDB()
        self.vector_store = VectorStore()

    def optimize(self):
        """
        اجرای جمع‌آوری زباله‌های حافظه (Garbage Collection) برای آزادسازی منابع.
        """
        gc.collect()

    def clear_unused_variables(self, variables: dict):
        """
        حذف متغیرهای استفاده‌نشده برای کاهش مصرف حافظه.
        """
        for var_name in list(variables.keys()):
            del variables[var_name]

    def log_memory_usage(self, model_id: str, used_memory: int):
        """
        ثبت میزان مصرف حافظه در TimescaleDB برای تحلیل‌های آینده.
        """
        timestamp = int(time.time())
        self.timescale_db.store_timeseries(
            metric="memory_usage",
            timestamp=timestamp,
            tags={"model_id": model_id},
            value=used_memory
        )

    def cache_heavy_computations(self, key: str, value: Any, ttl: int = 3600):
        """
        ذخیره محاسبات سنگین در Redis برای جلوگیری از اجرای مجدد پردازش‌های تکراری.
        """
        self.redis_cache.set_cache(key, value, ttl)

    def optimize_vector_storage(self, vector_id: str, vector_data: Any):
        """
        ذخیره و مدیریت بهینه بردارهای معنایی در Vector Store.
        """
        self.vector_store.store_vectors(vector_id, vector_data)

    def analyze_memory_trends(self):
        """
        تحلیل روند مصرف حافظه برای بهینه‌سازی آینده.
        """
        return self.timescale_db.query_timeseries(metric="memory_usage")
