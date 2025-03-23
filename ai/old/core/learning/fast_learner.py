from typing import Dict, Any, Optional, List, Set
from pathlib import Path
import logging
from datetime import datetime
import torch
from transformers import AutoModel, AutoTokenizer
from hazm import *
import numpy as np

from ..language.multilingual import MultilingualProcessor
from ..language.external.hazm_adapter import HazmAdapter
from ..language.external.parsbert_adapter import ParsBERTAdapter
from .core_learner import CoreLearner
from ..interfaces.exceptions import LearningError
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LearningProgress:
    """نگهداری وضعیت پیشرفت یادگیری"""
    vocabulary_size: int = 0
    patterns_learned: Set[str] = field(default_factory=set)
    embeddings_count: int = 0
    last_update: Optional[datetime] = None
    source_stats: Dict[str, int] = field(default_factory=dict)


class FastLearner:
    """یادگیرنده سریع از کتابخانه‌های موجود"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.knowledge_base_path = Path(self.config.get('knowledge_base_path', 'knowledge_base'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)

        # تنظیمات یادگیری
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.max_pattern_length = self.config.get('max_pattern_length', 5)
        self.min_frequency = self.config.get('min_frequency', 3)

        # اجزای اصلی
        self.mlp = MultilingualProcessor()
        self.hazm_adapter = HazmAdapter()
        self.parsbert = ParsBERTAdapter()

        # پیشرفت یادگیری
        self.progress = LearningProgress()
        self._initialize_models()

    async def _initialize_models(self):
        """راه‌اندازی مدل‌های پایه"""
        try:
            # Hazm
            self.normalizer = Normalizer()
            self.stemmer = Stemmer()
            self.lemmatizer = Lemmatizer()
            self.tagger = POSTagger(model='resources/postagger.model')
            self.chunker = Chunker(model='resources/chunker.model')

            # ParsBERT
            self.base_model = AutoModel.from_pretrained('HooshvareLab/bert-fa-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained('HooshvareLab/bert-fa-base-uncased')

            if torch.cuda.is_available():
                self.base_model = self.base_model.to('cuda')

            self.base_model.eval()
            logger.info("Models initialized successfully")

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    async def learn_language_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """یادگیری الگوهای زبانی از متون"""
        results = {
            'patterns': 0,
            'embeddings': 0,
            'concepts': 0
        }

        try:
            # پردازش با Hazm
            for text in texts:
                normalized = self.normalizer.normalize(text)
                words = word_tokenize(normalized)
                pos_tags = self.tagger.tag(words)
                chunks = self.chunker.parse(pos_tags)

                # استخراج الگوهای نحوی
                for chunk in chunks:
                    pattern = self._extract_pattern(chunk)
                    if pattern:
                        self.progress.patterns_learned.add(pattern)
                        results['patterns'] += 1

                # ساخت embedding با ParsBERT
                embedding = await self._create_embedding(normalized)
                if embedding is not None:
                    await self._store_embedding(normalized, embedding)
                    results['embeddings'] += 1

                # استخراج مفاهیم
                concepts = self._extract_concepts(pos_tags)
                self.progress.vocabulary_size += len(concepts)
                results['concepts'] += len(concepts)

            self.progress.last_update = datetime.now()
            return results

        except Exception as e:
            logger.error(f"Learning failed: {str(e)}")
            raise LearningError("Pattern learning failed") from e

    def _extract_pattern(self, chunk) -> Optional[str]:
        """استخراج الگوی نحوی از chunk"""
        if len(chunk) > self.max_pattern_length:
            return None

        pattern = []
        for word, tag in chunk:
            pattern.append(f"{tag}:{self.lemmatizer.lemmatize(word)}")
        return " ".join(pattern)

    async def _create_embedding(self, text: str) -> Optional[torch.Tensor]:
        """ساخت embedding برای متن"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.base_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).cpu()

        except Exception as e:
            logger.error(f"Embedding creation failed: {str(e)}")
            return None

    async def _store_embedding(self, text: str, embedding: torch.Tensor):
        """ذخیره embedding"""
        try:
            file_path = self.knowledge_base_path / f"embedding_{len(self.progress.patterns_learned)}.pt"
            data = {
                'text': text,
                'embedding': embedding,
                'timestamp': datetime.now().isoformat()
            }
            torch.save(data, file_path)
        except Exception as e:
            logger.error(f"Embedding storage failed: {str(e)}")

    def _extract_concepts(self, pos_tags: List[tuple]) -> Set[str]:
        """استخراج مفاهیم از متن"""
        concepts = set()
        for word, tag in pos_tags:
            if tag in ['N', 'ADJ', 'V']:
                lemma = self.lemmatizer.lemmatize(word)
                concepts.add(f"{tag}:{lemma}")
        return concepts

    def get_progress(self) -> Dict[str, Any]:
        """دریافت وضعیت پیشرفت یادگیری"""
        return {
            'vocabulary_size': self.progress.vocabulary_size,
            'patterns_count': len(self.progress.patterns_learned),
            'embeddings_count': self.progress.embeddings_count,
            'last_update': self.progress.last_update.isoformat() if self.progress.last_update else None,
            'source_stats': self.progress.source_stats
        }