import logging
from .models import AnalyticsQuery, AnalyticsResult

logger = logging.getLogger(__name__)

logger.info("Initializing ClickHouse Domain Module...")

__all__ = [
    "AnalyticsQuery",
    "AnalyticsResult"
]
