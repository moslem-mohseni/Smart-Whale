"""
chat_embeddings.py - Schema for chat message embeddings in Milvus

This collection stores vector embeddings for chat messages,
enabling semantic search and context-aware responses across conversations.
"""

from storage.scripts.base_schema import MilvusSchema
from pymilvus import DataType


class ChatEmbeddingsCollection(MilvusSchema):
    """
    Collection schema for chat message embeddings
    """

    @property
    def name(self) -> str:
        return "chat_embeddings"

    @property
    def description(self) -> str:
        return "Vector embeddings for chat messages"

    def get_create_statement(self) -> dict:
        """Define Milvus collection schema"""
        return {
            "fields": [
                {"name": "id", "type": DataType.INT64, "is_primary": True},
                {"name": "embedding", "type": DataType.FLOAT_VECTOR, "dim": 1536},
                {"name": "message_id", "type": DataType.VARCHAR, "max_length": 100},
                {"name": "chat_id", "type": DataType.VARCHAR, "max_length": 100},
                {"name": "content", "type": DataType.VARCHAR, "max_length": 4096},
                {"name": "role", "type": DataType.VARCHAR, "max_length": 50},
                {"name": "created_at", "type": DataType.INT64},
                {"name": "user_id", "type": DataType.INT64}
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
