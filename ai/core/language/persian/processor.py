"""Persian Language Processor"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import asyncio
import numpy as np

from ..base.language_processor import LanguageProcessor, ProcessingResult
from ..external.parsbert_adapter import ParsBERTAdapter, ParsBERTResult
from ..external.hazm_adapter import HazmAdapter, HazmResult

logger = logging.getLogger(__name__)

class PersianProcessor(LanguageProcessor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.parsbert = ParsBERTAdapter()
        self.hazm = HazmAdapter()
        self.min_confidence = config.get('min_confidence', 0.7)
        self._process_lock = asyncio.Lock()
        self._learning_history = []

    async def initialize(self) -> bool:
        try:
            parsbert_init = await self.parsbert.initialize()
            hazm_init = await self.hazm.initialize()

            if not parsbert_init or not hazm_init:
                logger.error("Failed to initialize adapters")
                return False

            self._initialized = True
            logger.info("Persian processor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False

    async def process(self, text: str) -> ProcessingResult:
        async with self._process_lock:
            if not await self.validate_text(text):
                raise ValueError("Invalid text")

            parsbert_result = await self.parsbert.process_text(text)
            hazm_result = await self.hazm.process_text(text)

            if not parsbert_result or not hazm_result:
                raise RuntimeError("Processing failed")

            combined_features = self._combine_analyses(parsbert_result, hazm_result)
            confidence = self._calculate_confidence(parsbert_result, hazm_result)

            return ProcessingResult(
                text=text,
                tokens=hazm_result.words,
                confidence=confidence,
                language='fa',
                analysis_time=datetime.now(),
                features=combined_features,
                metadata={
                    'normalized': hazm_result.normalized_text,
                    'sentences': len(hazm_result.sentences),
                    'words': len(hazm_result.words)
                }
            )

    async def learn(self, text: str, analysis: ProcessingResult) -> None:
        """یادگیری از نتایج تحلیل متن"""
        try:
            hazm_result = await self.hazm.process_text(text)
            parsbert_result = await self.parsbert.process_text(text)

            if not hazm_result or not parsbert_result:
                logger.warning("Learning failed - processing error")
                return

            learning_data = {
                'text': text,
                'features': analysis.features,
                'confidence': analysis.confidence,
                'timestamp': datetime.now(),
                'parsbert_embedding': parsbert_result.sentence_embedding.numpy(),
                'hazm_analysis': {
                    'pos_tags': hazm_result.pos_tags,
                    'lemmas': hazm_result.lemmas
                }
            }

            self._learning_history.append(learning_data)
            await self._update_knowledge(learning_data)

            logger.info("Learning completed successfully")
        except Exception as e:
            logger.error(f"Learning error: {e}")

    def _combine_analyses(self, parsbert: ParsBERTResult, hazm: HazmResult) -> Dict[str, Any]:
        return {
            'semantic': {
                'embeddings': parsbert.embeddings.numpy(),
                'sentence_vector': parsbert.sentence_embedding.numpy()
            },
            'structural': {
                'lemmas': hazm.lemmas,
                'pos_tags': hazm.pos_tags,
                'chunks': hazm.chunks
            },
            'attention': {
                'weights': parsbert.attention_weights.numpy() if parsbert.attention_weights is not None else None
            }
        }

    def _calculate_confidence(self, parsbert: ParsBERTResult, hazm: HazmResult) -> float:
        weights = {'semantic': 0.6, 'structural': 0.4}
        semantic_conf = float(parsbert.attention_weights.max()) if parsbert.attention_weights is not None else 0.5
        structural_conf = sum(1 for tag in hazm.pos_tags if tag[1] != 'UNKNOWN') / len(hazm.pos_tags)
        return weights['semantic'] * semantic_conf + weights['structural'] * structural_conf

    async def validate_text(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        persian_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return persian_chars / len(text) >= 0.5

    async def _update_knowledge(self, learning_data: Dict[str, Any]) -> None:
        """به‌روزرسانی پایگاه دانش"""
        if len(self._learning_history) > 1000:
            self._learning_history = self._learning_history[-1000:]

    async def analyze_deeply(self, text: str) -> Optional[ProcessingResult]:
        result = await self.process(text)
        if result.confidence < self.min_confidence:
            enriched_result = await self._enrich_analysis(result)
            return enriched_result if enriched_result else result
        return result

    async def _enrich_analysis(self, result: ProcessingResult) -> Optional[ProcessingResult]:
        """غنی‌سازی تحلیل با استفاده از تاریخچه یادگیری"""
        return result

    async def _get_external_analysis(self, text: str) -> Optional[ProcessingResult]:
        return None