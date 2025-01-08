# infrastructure/clickhouse/__init__.py
"""
ClickHouse Service
----------------
Analytics database for high-performance data analysis.
Handles large-scale data processing and analytical queries.
"""

from .service.analytics_service import AnalyticsService
from .domain.models import (
    AnalyticsEvent,
    AnalyticsQuery,
    AnalyticsResult,
    TableSchema
)
from .config.settings import (
    ClickHouseConfig,
    QuerySettings
)
from .scripts.maintenance import MaintenanceManager

__version__ = '1.0.0'

__all__ = [
    # Core Services
    'AnalyticsService',
    'MaintenanceManager',

    # Domain Models
    'AnalyticsEvent',
    'AnalyticsQuery',
    'AnalyticsResult',
    'TableSchema',

    # Configurations
    'ClickHouseConfig',
    'QuerySettings'
]