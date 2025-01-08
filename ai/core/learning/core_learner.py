from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import json
import logging
from pathlib import Path

from ..knowledge.knowledge_manager import KnowledgeManager
from ..metrics.metrics_collector import MetricsCollector, LearningMetric
from ..errors.exceptions import LearningError

logger = logging.getLogger(__name__)


class CoreLearner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.knowledge_manager = KnowledgeManager()
        self.metrics_collector = MetricsCollector(config.get('metrics', {}))
        self.knowledge_base_path = Path(config.get('knowledge_base_path', 'knowledge_base'))
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self._learning_state = {}

    async def initialize(self) -> bool:
        """راه‌اندازی سیستم یادگیری"""
        try:
            await self.knowledge_manager.initialize()
            await self.metrics_collector.start_collection()
            await self._load_learning_state()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize core learner: {e}")
            return False

    async def learn(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """پردازش و یادگیری محتوای جدید"""
        start_time = datetime.now()
        try:
            # استخراج مفاهیم و دانش از محتوا
            knowledge = await self._extract_knowledge(content, context)

            # ذخیره در پایگاه دانش
            knowledge_id = await self._store_knowledge(knowledge)

            # به‌روزرسانی وضعیت یادگیری
            self._update_learning_state(knowledge)

            # ثبت متریک‌ها
            self._record_learning_metrics(content, knowledge, start_time)

            return {
                'knowledge_id': knowledge_id,
                'concepts_learned': knowledge['concepts'],
                'confidence_score': knowledge['confidence']
            }

        except Exception as e:
            logger.error(f"Learning failed: {e}")
            raise LearningError(str(e), query=content)

    async def _extract_knowledge(self, content: str, context: Optional[Dict]) -> Dict[str, Any]:
        """استخراج دانش از محتوا"""
        try:
            # استفاده از منابع دانش برای استخراج اطلاعات
            results = await self.knowledge_manager.learn(content, context)

            # ترکیب و پردازش نتایج
            processed_results = await self._process_results(results)

            # استخراج مفاهیم کلیدی
            concepts = await self._extract_concepts(processed_results)

            return {
                'content': content,
                'processed_content': processed_results,
                'concepts': concepts,
                'context': context,
                'confidence': self._calculate_confidence(processed_results),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            raise

    async def _process_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """پردازش و ترکیب نتایج از منابع مختلف"""
        processed = {
            'main_concepts': set(),
            'relationships': [],
            'metadata': {}
        }

        for result in results:
            content = result.get('content', '')
            metadata = result.get('metadata', {})

            # استخراج مفاهیم اصلی
            concepts = await self._extract_main_concepts(content)
            processed['main_concepts'].update(concepts)

            # استخراج روابط
            relationships = await self._extract_relationships(content, concepts)
            processed['relationships'].extend(relationships)

            # ترکیب متادیتا
            self._merge_metadata(processed['metadata'], metadata)

        return processed

    async def _extract_main_concepts(self, content: str) -> set:
        """استخراج مفاهیم اصلی از متن"""
        # در نسخه‌های بعدی پیاده‌سازی می‌شود
        return set()

    async def _extract_relationships(self, content: str, concepts: set) -> List[Dict]:
        """استخراج روابط بین مفاهیم"""
        # در نسخه‌های بعدی پیاده‌سازی می‌شود
        return []

    def _merge_metadata(self, base_metadata: Dict, new_metadata: Dict) -> None:
        """ترکیب متادیتای جدید با متادیتای موجود"""
        for key, value in new_metadata.items():
            if key in base_metadata:
                if isinstance(value, dict):
                    self._merge_metadata(base_metadata[key], value)
                elif isinstance(value, list):
                    base_metadata[key] = list(set(base_metadata[key] + value))
                else:
                    base_metadata[key] = value
            else:
                base_metadata[key] = value

    async def _store_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """ذخیره دانش در پایگاه دانش"""
        try:
            knowledge_id = self._generate_knowledge_id(knowledge)
            file_path = self.knowledge_base_path / f"{knowledge_id}.json"

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(knowledge, ensure_ascii=False, indent=2))

            return knowledge_id

        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            raise

    def _generate_knowledge_id(self, knowledge: Dict[str, Any]) -> str:
        """تولید شناسه یکتا برای دانش"""
        content = knowledge.get('content', '')
        timestamp = knowledge.get('timestamp', datetime.now()).isoformat()
        return hashlib.sha256(f"{content}{timestamp}".encode()).hexdigest()[:12]

    def _update_learning_state(self, knowledge: Dict[str, Any]) -> None:
        """به‌روزرسانی وضعیت یادگیری"""
        concepts = knowledge.get('concepts', [])
        for concept in concepts:
            if concept not in self._learning_state:
                self._learning_state[concept] = {
                    'first_seen': datetime.now(),
                    'occurrence_count': 1,
                    'confidence': knowledge['confidence']
                }
            else:
                self._learning_state[concept]['occurrence_count'] += 1
                self._learning_state[concept]['confidence'] = max(
                    self._learning_state[concept]['confidence'],
                    knowledge['confidence']
                )

    def _record_learning_metrics(self, content: str, knowledge: Dict[str, Any],
                                 start_time: datetime) -> None:
        """ثبت متریک‌های یادگیری"""
        metric = LearningMetric(
            query_time=(datetime.now() - start_time).total_seconds(),
            source_count=len(knowledge.get('processed_content', {}).get('metadata', {})),
            success_rate=1.0,  # موفق
            token_count=len(content.split()),
            confidence_score=knowledge['confidence'],
            response_length=len(json.dumps(knowledge)),
            timestamp=datetime.now()
        )
        self.metrics_collector.record_learning_metric(metric)

    async def _load_learning_state(self) -> None:
        """بارگذاری وضعیت یادگیری از فایل"""
        state_file = self.knowledge_base_path / 'learning_state.json'
        if state_file.exists():
            try:
                async with aiofiles.open(state_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    self._learning_state = json.loads(content)
            except Exception as e:
                logger.error(f"Failed to load learning state: {e}")
                self._learning_state = {}

    async def save_learning_state(self) -> None:
        """ذخیره وضعیت یادگیری در فایل"""
        state_file = self.knowledge_base_path / 'learning_state.json'
        try:
            async with aiofiles.open(state_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self._learning_state, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")

    async def cleanup(self) -> None:
        """پاکسازی منابع"""
        await self.save_learning_state()
        await self.metrics_collector.stop_collection()
        await self.knowledge_manager.cleanup()
        logger.info("Core learner cleaned up")