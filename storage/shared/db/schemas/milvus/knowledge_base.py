"""
knowledge_base.py - Schema for knowledge base embeddings in Milvus

This collection stores vector embeddings for the shared knowledge base,
enabling semantic search across different models and components.
"""

from storage.scripts.base_schema import MilvusSchema
from pymilvus import DataType


class KnowledgeBaseCollection(MilvusSchema):
    """
    Collection schema for knowledge base embeddings
    """

    @property
    def name(self) -> str:
        return "knowledge_base"

    @property
    def description(self) -> str:
        return "Vector embeddings for shared knowledge base"

    def get_create_statement(self) -> dict:
        """Define Milvus collection schema"""
        return {
            "fields": [
                {"name": "id", "type": DataType.INT64, "is_primary": True},
                {"name": "embedding", "type": DataType.FLOAT_VECTOR, "dim": 1536},
                {"name": "content", "type": DataType.VARCHAR, "max_length": 4096},
                {"name": "source", "type": DataType.VARCHAR, "max_length": 255},
                {"name": "category", "type": DataType.VARCHAR, "max_length": 100},
                {"name": "created_at", "type": DataType.INT64},
                {"name": "metadata", "type": DataType.JSON}
            ],
            "index_params": {
                "field_name": "embedding",
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            },
            "search_params": {
                "metric_type": "COSINE",
                "params": {"ef": 128}
            },
            "consistency_level": "Strong"
        }

    def get_check_exists_statement(self) -> str:
        """Collection name to check for existence"""
        return self.name
    