"""
ماژول `context/` مسئول مدیریت حافظه زمینه‌ای، تحلیل و پردازش زمینه‌ی مکالمات، و مدیریت اطلاعات بلندمدت برای افزایش دقت پردازش زبان طبیعی (NLP) است.

این ماژول شامل چهار بخش اصلی است:
- `memory/` : مدیریت حافظه و کش چندسطحی برای پردازش سریع‌تر داده‌های مکالمه‌ای
- `analyzer/` : تحلیل و بررسی معنایی مکالمات برای درک عمیق‌تر زمینه گفتگو
- `manager/` : رهگیری مکالمات، مدیریت نشست‌ها، تعیین وضعیت مکالمه و بروزرسانی داده‌ها
- `retriever/` : بازیابی اطلاعات از کش و حافظه و ترکیب داده‌های مکالمه‌ای برای پاسخ‌های دقیق‌تر
"""

from .memory import (
    L1Cache,
    L2Cache,
    L3Cache,
    CacheSynchronizer,
    QuantumMemory,
    QuantumCompressor,
    RetrievalOptimizer,
)

from .analyzer import (
    AdaptiveLearning,
    ContextDetector,
    HistoryAnalyzer,
    NoiseFilter,
    RelevanceChecker,
    SemanticAnalyzer,
    Summarizer,
    TopicStore,
    VectorSearch,
)

from .manager import (
    ContextTracker,
    SessionHandler,
    StateManager,
    FallbackHandler,
    UpdatePolicy,
)

from .retriever import (
    CacheLookup,
    DataAggregator,
    FactChecker,
    KnowledgeGraph,
    RetrieverVectorSearch,
)

__all__ = [
    "L1Cache",
    "L2Cache",
    "L3Cache",
    "CacheSynchronizer",
    "QuantumMemory",
    "QuantumCompressor",
    "RetrievalOptimizer",
    "ContextTracker",
    "SessionHandler",
    "StateManager",
    "FallbackHandler",
    "UpdatePolicy",
    "CacheLookup",
    "DataAggregator",
    "FactChecker",
    "KnowledgeGraph",
    "RetrieverVectorSearch",
    "AdaptiveLearning",
    "ContextDetector",
    "HistoryAnalyzer",
    "NoiseFilter",
    "RelevanceChecker",
    "SemanticAnalyzer",
    "Summarizer",
    "TopicStore",
    "VectorSearch",
]
