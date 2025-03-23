from infrastructure.redis.config.settings import RedisConfig
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from infrastructure.redis.service.cache_service import CacheService
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.clickhouse.service.analytics_service import AnalyticsService
from infrastructure.interfaces.storage import StorageInterface
from infrastructure.clickhouse.domain.models import AnalyticsEvent
from .exceptions import CollectionError, ProcessingError, StorageError


class BaseCollector(ABC):
    """کلاس پایه برای تمام collectors"""

    def __init__(
            self,
            source_name: str,
            batch_size: int = 100,
            cache_service: Optional[CacheService] = None,
            kafka_service: Optional[KafkaService] = None,
            analytics_service: Optional[AnalyticsService] = None,
            storage: Optional[StorageInterface] = None
    ):
        self.source_name = source_name
        self.batch_size = batch_size

        redis_config = RedisConfig.from_env()
        self.cache_service = cache_service or CacheService(redis_config)
        self.kafka_service = kafka_service or KafkaService()
        self.analytics_service = analytics_service or AnalyticsService()
        self.storage = storage

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{source_name}")
        self._setup_logger()

        asyncio.create_task(self._initialize_services())

    async def process_batch(self, batch: List[Dict[str, Any]]) -> None:
        try:
            cache_key = f"{self.source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.cache_service.set(cache_key, batch)

            topic = f"{self.source_name}_data"
            for item in batch:
                await self.kafka_service.send_message(topic, item)

            event = AnalyticsEvent(
                event_id=f"{self.source_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                event_type="batch_processing",
                batch_size=int(len(batch)),
                source=self.source_name,
                timestamp=datetime.now().isoformat()
            )
            await self.analytics_service.store_event(event)

            self.logger.info(f"پردازش موفق {len(batch)} آیتم از {self.source_name}")

        except Exception as e:
            self.logger.error(f"خطا در پردازش batch از {self.source_name}: {str(e)}")
            raise ProcessingError(str(e))
