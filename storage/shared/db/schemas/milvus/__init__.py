"""
milvus/__init__.py - Milvus Collection Schemas for Shared Components

This module imports all Milvus collection schemas used for shared vector search
functionality across different language models.
"""

from .knowledge_base import KnowledgeBaseCollection
from .chat_embeddings import ChatEmbeddingsCollection

__all__ = [
    'KnowledgeBaseCollection',
    'ChatEmbeddingsCollection'
]
