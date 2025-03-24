"""
ماژول `vector_store/` وظیفه‌ی مدیریت بردارهای پردازشی زبان را بر عهده دارد.

📌 اجزای اصلی این ماژول:
- `vector_search.py` → جستجوی برداری در داده‌های پردازشی زبان
- `milvus_adapter.py` → مدیریت ارتباط با Milvus و انجام عملیات برداری
"""

from .vector_search import VectorSearch
from .milvus_adapter import MilvusAdapter
from infrastructure.vector_store.service.vector_service import VectorService

# مقداردهی اولیه MilvusAdapter با استفاده از سرویس `VectorService`
vector_service = VectorService()
milvus_adapter = MilvusAdapter(vector_service)

# مقداردهی اولیه VectorSearch با استفاده از MilvusAdapter
vector_search = VectorSearch(milvus_adapter)

__all__ = [
    "vector_search",
    "milvus_adapter",
    "VectorSearch",
    "MilvusAdapter",
]
