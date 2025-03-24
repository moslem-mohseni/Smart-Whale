from pymilvus import Collection
from ..adapters import ConnectionPool, RetryMechanism
from ..domain.collection import MilvusCollectionManager
from ..config.config import config as index_config


class IndexService:
    """سرویس مدیریت ایندکس‌ها در Milvus"""

    def __init__(self, collection_name: str = None):
        self.collection_manager = MilvusCollectionManager(collection_name)
        self.connection_pool = ConnectionPool()
        self.retry = RetryMechanism()

    @RetryMechanism().retry
    def create_index(self):
        """ایجاد ایندکس روی Collection برای بهینه‌سازی جستجو"""
        collection = Collection(self.collection_manager.collection_name)
        index_params = {
            "index_type": index_config.INDEX_TYPE,
            "metric_type": "L2",
            "params": {
                "nlist": index_config.IVF_NLIST if "IVF" in index_config.INDEX_TYPE else None,
                "M": index_config.HNSW_M if "HNSW" in index_config.INDEX_TYPE else None,
                "efConstruction": index_config.HNSW_EF if "HNSW" in index_config.INDEX_TYPE else None,
            }
        }
        index_params["params"] = {k: v for k, v in index_params["params"].items() if v is not None}  # حذف مقادیر None
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"✅ ایندکس '{index_config.INDEX_TYPE}"
              f"' برای Collection '{self.collection_manager.collection_name}' ایجاد شد.")

    @RetryMechanism().retry
    def drop_index(self):
        """حذف ایندکس از Collection"""
        collection = Collection(self.collection_manager.collection_name)
        collection.drop_index()
        print(f"🗑️ ایندکس از Collection '{self.collection_manager.collection_name}' حذف شد.")

    @RetryMechanism().retry
    def rebuild_index(self):
        """بازسازی ایندکس در صورت تغییرات داده‌ها"""
        self.drop_index()
        self.create_index()
        print(f"🔄 ایندکس Collection '{self.collection_manager.collection_name}' بازسازی شد.")
