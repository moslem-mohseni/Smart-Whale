# infrastructure/vector_store/monitoring/health_check.py

from pymilvus import utility, connections, Collection
import logging
from .metrics import metrics

logger = logging.getLogger(__name__)


class HealthCheck:
    """بررسی سلامت Vector Store و نظارت بر وضعیت سیستم"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def check_connection(self):
        """بررسی وضعیت اتصال به Vector Store"""
        try:
            connected = connections.has_connection("default")
            metrics.set_connection_status(connected)
            logger.info(f"وضعیت Vector Store: {'متصل' if connected else 'قطع'}")
            return connected
        except Exception as e:
            metrics.set_connection_status(False)
            logger.error(f"خطا در بررسی اتصال Vector Store: {str(e)}")
            return False

    def check_collection_health(self):
        """بررسی وضعیت Collection و تعداد بردارها"""
        try:
            if not utility.has_collection(self.collection_name):
                logger.warning(f"Collection '{self.collection_name}' وجود ندارد.")
                return False

            collection = Collection(self.collection_name)
            total_entities = collection.num_entities
            logger.info(f"Collection '{self.collection_name}' سالم است و دارای {total_entities} بردار است.")
            return True
        except Exception as e:
            logger.error(f"خطا در بررسی Collection '{self.collection_name}': {str(e)}")
            return False
