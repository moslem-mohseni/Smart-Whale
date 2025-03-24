"""
ماژول `retriever/` مسئول بازیابی و پردازش اطلاعات مکالمه‌ای از منابع مختلف است.

این ماژول شامل بخش‌های زیر است:
- `cache_lookup`: بررسی کش برای داده‌های پرتکرار
- `data_aggregator`: ترکیب داده‌ها از منابع مختلف
- `fact_checker`: بررسی صحت داده‌ها
- `knowledge_graph`: نمایش داده‌ها در قالب یک نمودار دانش
- `vector_search`: جستجوی برداری برای یافتن اطلاعات مرتبط
"""

from .cache_lookup import CacheLookup
from .data_aggregator import DataAggregator
from .fact_checker import FactChecker
from .knowledge_graph import KnowledgeGraph
from .vector_search import RetrieverVectorSearch

# مقداردهی اولیه ماژول‌ها
cache_lookup = CacheLookup()
data_aggregator = DataAggregator()
fact_checker = FactChecker()
knowledge_graph = KnowledgeGraph()
vector_search = RetrieverVectorSearch()

__all__ = [
    "CacheLookup",
    "DataAggregator",
    "FactChecker",
    "KnowledgeGraph",
    "RetrieverVectorSearch",
]
