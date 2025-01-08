# infrastructure/timescaledb/__init__.py
"""
TimescaleDB Service
-----------------
Time-series database for storing and analyzing financial and system data.
Includes database migrations and management scripts.
"""

from .service.database_service import TimescaleDBService
from .adapters.repository import Repository
from .adapters.timeseries_repository import TimeSeriesRepository
from .domain.models import TimeSeriesData, TableSchema
from .domain.value_objects import TimeRange
from .config.settings import TimescaleDBConfig
from .storage.timescaledb_storage import TimescaleDBStorage

__all__ = [
    'TimescaleDBService',
    'Repository',
    'TimeSeriesRepository',
    'TimeSeriesData',
    'TableSchema',
    'TimescaleDBConfig',
    'TimeRange',
    'TimescaleDBStorage'
]