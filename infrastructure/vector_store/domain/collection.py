from typing import List, Dict, Any
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
from .models import Vector
from ..config.config import config as collection_config


class MilvusCollectionManager:
    """مدیریت Collectionها در Milvus"""

    def __init__(self, collection_name: str = None):
        """
        مقداردهی اولیه کلاس مدیریت Collection
        :param collection_name: نام Collection (در صورت عدم مقداردهی، مقدار پیش‌فرض تنظیمات استفاده می‌شود)
        """
        self.collection_name = collection_name or collection_config.DEFAULT_COLLECTION_NAME
        self.collection = None

    def create_collection(self):
        """ایجاد Collection در Milvus (در صورتی که وجود نداشته باشد)"""
        if utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' از قبل وجود دارد.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=collection_config.VECTOR_DIMENSIONS),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(fields=fields, description="Collection for storing vectors")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"Collection '{self.collection_name}' با موفقیت ایجاد شد.")

    def drop_collection(self):
        """حذف Collection از Milvus"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' حذف شد.")
        else:
            print(f"Collection '{self.collection_name}' وجود ندارد.")

    def insert_vectors(self, vectors: List[Vector]):
        """افزودن بردارها به Collection"""
        if not self.collection:
            self.collection = Collection(self.collection_name)

        data = [[v.id for v in vectors], [v.values for v in vectors], [v.metadata for v in vectors]]
        self.collection.insert(data)
        print(f"{len(vectors)} بردار به Collection '{self.collection_name}' اضافه شد.")

    def delete_vectors(self, ids: List[str]):
        """حذف بردارها از Collection بر اساس ID"""
        if not self.collection:
            self.collection = Collection(self.collection_name)

        expr = f"id in {ids}"
        self.collection.delete(expr)
        print(f"{len(ids)} بردار از Collection '{self.collection_name}' حذف شد.")

    def has_collection(self) -> bool:
        """بررسی وجود Collection در Milvus"""
        return utility.has_collection(self.collection_name)

    def load_collection(self):
        """بارگذاری Collection در حافظه برای جستجوهای سریع‌تر"""
        if not self.collection:
            self.collection = Collection(self.collection_name)

        self.collection.load()
        print(f"Collection '{self.collection_name}' در حافظه بارگذاری شد.")
