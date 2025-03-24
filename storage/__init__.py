"""
ماژول پایگاه داده که شامل طرح‌های مختلف پایگاه‌های داده و واسط‌های ارتباطی آنهاست.
این ماژول امکان دسترسی به طرح‌های عمومی و طرح‌های ویژه زبان فارسی را فراهم می‌کند.
"""

# واردسازی طرح‌های عمومی
from .shared.db.schemas import clickhouse as shared_clickhouse
from .shared.db.schemas import milvus as shared_milvus
from .shared.db.schemas import timescaledb as shared_timescaledb

# واردسازی طرح‌های ویژه زبان فارسی
from .models.language.persian.db.schemas import clickhouse as persian_clickhouse
from .models.language.persian.db.schemas import milvus as persian_milvus
from .models.language.persian.db.schemas import timescaledb as persian_timescaledb

clickhouse = shared_clickhouse
milvus = shared_milvus
timescaledb = shared_timescaledb

persian_schemas = {
    'clickhouse': persian_clickhouse,
    'milvus': persian_milvus,
    'timescaledb': persian_timescaledb
}

__all__ = [
    'clickhouse',
    'milvus',
    'timescaledb',
    'persian_schemas'
]