import asyncio
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
import logging

from .interfaces.knowledge_source import (
    KnowledgeSource, KnowledgeSourceType, LearningContext, KnowledgeMetadata
)
from .sources.chatgpt_source import ChatGPTSource
from .sources.wikipedia_source import WikipediaSource
from .sources.local_file_source import LocalFileSource

logger = logging.getLogger(__name__)


class KnowledgeManager:
    def __init__(self):
        self._sources: Dict[str, KnowledgeSource] = {}
        self._source_configs: Dict[str, Dict[str, Any]] = {}
        self._learning_history: List[Dict[str, Any]] = []
        self._initialized = False

        # نگاشت نوع منبع به کلاس مربوطه
        self._source_types = {
            KnowledgeSourceType.CHAT_GPT: ChatGPTSource,
            KnowledgeSourceType.WIKIPEDIA: WikipediaSource,
            KnowledgeSourceType.LOCAL_FILE: LocalFileSource
        }

    async def initialize(self) -> bool:
        """راه‌اندازی مدیر دانش و تمام منابع آن"""
        try:
            for source_id, config in self._source_configs.items():
                await self._initialize_source(source_id, config)
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize knowledge manager: {str(e)}")
            return False

    async def _initialize_source(self, source_id: str, config: Dict[str, Any]) -> None:
        """راه‌اندازی یک منبع دانش خاص"""
        source_type = KnowledgeSourceType(config['type'])
        source_class = self._source_types.get(source_type)

        if not source_class:
            raise ValueError(f"Unknown source type: {source_type}")

        source = source_class(config)
        success = await source.initialize()

        if success:
            self._sources[source_id] = source
        else:
            logger.error(f"Failed to initialize source: {source_id}")

    async def add_knowledge_source(self, source_id: str, config: Dict[str, Any]) -> bool:
        """افزودن یک منبع دانش جدید"""
        if source_id in self._sources:
            logger.warning(f"Source {source_id} already exists")
            return False

        self._source_configs[source_id] = config
        if self._initialized:
            await self._initialize_source(source_id, config)
        return True

    async def remove_knowledge_source(self, source_id: str) -> bool:
        """حذف یک منبع دانش"""
        if source_id not in self._sources:
            return False

        source = self._sources[source_id]
        await source.cleanup()
        del self._sources[source_id]
        del self._source_configs[source_id]
        return True

    async def learn(self, query: str, context: Optional[LearningContext] = None,
                    source_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """یادگیری از منابع دانش مختلف"""
        if not self._initialized:
            raise RuntimeError("Knowledge manager is not initialized")

        results = []
        sources_to_query = (
            [self._sources[sid] for sid in source_ids if sid in self._sources]
            if source_ids
            else self._sources.values()
        )

        tasks = [
            self._learn_from_source(source, query, context)
            for source in sources_to_query
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for response in responses:
            if isinstance(response, Exception):
                logger.error(f"Error in learning: {str(response)}")
                continue
            if response:
                results.append(response)
                self._learning_history.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'response': response
                })

        return results

    async def _learn_from_source(self, source: KnowledgeSource,
                                 query: str,
                                 context: Optional[LearningContext]) -> Optional[Dict[str, Any]]:
        """یادگیری از یک منبع خاص"""
        try:
            response = await source.get_knowledge(query, context)
            if not await source.validate_knowledge(response):
                logger.warning(f"Invalid knowledge from source {source.__class__.__name__}")
                return None
            return response
        except Exception as e:
            logger.error(f"Error learning from source {source.__class__.__name__}: {str(e)}")
            return None

    def get_learning_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """دریافت تاریخچه یادگیری"""
        history = self._learning_history
        if limit:
            history = history[-limit:]
        return history

    async def validate_all_sources(self) -> Dict[str, bool]:
        """اعتبارسنجی تمام منابع دانش"""
        results = {}
        for source_id, source in self._sources.items():
            try:
                # ارسال یک درخواست ساده برای تست
                response = await source.get_knowledge("test query")
                results[source_id] = bool(response)
            except Exception as e:
                logger.error(f"Source validation failed for {source_id}: {str(e)}")
                results[source_id] = False
        return results

    async def cleanup(self) -> None:
        """پاکسازی تمام منابع"""
        cleanup_tasks = [
            source.cleanup()
            for source in self._sources.values()
        ]
        await asyncio.gather(*cleanup_tasks)
        self._sources.clear()
        self._initialized = False