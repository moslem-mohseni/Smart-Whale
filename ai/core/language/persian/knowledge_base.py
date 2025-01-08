"""Persian Knowledge Base"""
from typing import Dict, List, Any, Optional, Set
import logging
from datetime import datetime
import numpy as np
from pathlib import Path
import json
import asyncio
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SemanticConcept:
    """مفهوم معنایی"""
    word: str
    vector: np.ndarray
    synonyms: Set[str] = field(default_factory=set)
    antonyms: Set[str] = field(default_factory=set)
    related: Dict[str, float] = field(default_factory=dict)
    usage_count: int = field(default=0)
    last_used: Optional[datetime] = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'word': self.word,
            'vector': self.vector.tolist(),
            'synonyms': list(self.synonyms),
            'antonyms': list(self.antonyms),
            'related': self.related,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }


@dataclass
class GrammaticalPattern:
    """الگوی دستوری"""
    pattern: str
    examples: List[str] = field(default_factory=list)
    frequency: int = field(default=0)
    confidence: float = field(default=0.0)
    last_seen: Optional[datetime] = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern': self.pattern,
            'examples': self.examples,
            'frequency': self.frequency,
            'confidence': self.confidence,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None
        }


class PersianKnowledgeBase:
    """پایگاه دانش زبان فارسی"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.storage_path = Path(self.config.get('storage_path', 'knowledge'))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.concepts: Dict[str, SemanticConcept] = {}
        self.patterns: Dict[str, GrammaticalPattern] = {}

        self._concept_lock = asyncio.Lock()
        self._pattern_lock = asyncio.Lock()

        self._stats = {
            'total_concepts': 0,
            'total_patterns': 0,
            'total_updates': 0,
            'last_update': None
        }

    async def add_concept(self, word: str, vector: np.ndarray,
                          related_words: Optional[Dict[str, float]] = None) -> None:
        async with self._concept_lock:
            if word in self.concepts:
                concept = self.concepts[word]
                concept.vector = self._update_vector(concept.vector, vector)
                if related_words:
                    concept.related.update(related_words)
                concept.usage_count += 1
                concept.last_used = datetime.now()
            else:
                self.concepts[word] = SemanticConcept(
                    word=word,
                    vector=vector,
                    related=related_words or {}
                )

            self._update_stats()

    async def add_pattern(self, pattern: str, example: str, confidence: float = 1.0) -> None:
        async with self._pattern_lock:
            if pattern in self.patterns:
                p = self.patterns[pattern]
                p.examples.append(example)
                p.frequency += 1
                p.confidence = self._update_confidence(p.confidence, confidence)
                p.last_seen = datetime.now()
            else:
                self.patterns[pattern] = GrammaticalPattern(
                    pattern=pattern,
                    examples=[example],
                    frequency=1,
                    confidence=confidence
                )

            self._update_stats()

    def _update_stats(self) -> None:
        self._stats.update({
            'total_concepts': len(self.concepts),
            'total_patterns': len(self.patterns),
            'total_updates': self._stats['total_updates'] + 1,
            'last_update': datetime.now()
        })

    def _serialize_stats(self) -> Dict[str, Any]:
        try:
            last_update = self._stats.get('last_update')
            if isinstance(last_update, datetime):
                last_update_str = last_update.isoformat()
            else:
                last_update_str = None

            return {
                'total_concepts': self._stats['total_concepts'],
                'total_patterns': self._stats['total_patterns'],
                'total_updates': self._stats['total_updates'],
                'last_update': last_update_str
            }
        except Exception as e:
            logger.error(f"Error serializing stats: {e}")
            return {}

    async def save_knowledge(self) -> None:
        try:
            save_data = {
                'concepts': {
                    word: concept.to_dict()
                    for word, concept in self.concepts.items()
                },
                'patterns': {
                    pattern.pattern: pattern.to_dict()
                    for pattern in self.patterns.values()
                },
                'stats': self._serialize_stats()
            }

            file_path = self.storage_path / 'persian_knowledge.json'
            async with asyncio.Lock():
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)

            logger.info("Knowledge base saved successfully")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")

    async def find_matching_patterns(self, text: str,
                                     min_confidence: float = 0.5) -> List[GrammaticalPattern]:
        """جستجوی الگوهای منطبق با متن"""
        matching = []
        for pattern in self.patterns.values():
            if (pattern.confidence >= min_confidence and
                    any(example in text for example in pattern.examples)):
                matching.append(pattern)

        return sorted(matching, key=lambda p: p.frequency, reverse=True)

    async def load_knowledge(self) -> None:
        try:
            file_path = self.storage_path / 'persian_knowledge.json'
            if not file_path.exists():
                logger.info("No existing knowledge base found")
                return

            async with asyncio.Lock():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            # Load concepts
            for word, info in data.get('concepts', {}).items():
                self.concepts[word] = SemanticConcept(
                    word=word,
                    vector=np.array(info['vector']),
                    synonyms=set(info.get('synonyms', [])),
                    antonyms=set(info.get('antonyms', [])),
                    related=info.get('related', {}),
                    usage_count=info.get('usage_count', 0),
                    last_used=datetime.fromisoformat(info['last_used']) if info.get('last_used') else None
                )

            # Load patterns
            for pattern_str, info in data.get('patterns', {}).items():
                self.patterns[pattern_str] = GrammaticalPattern(
                    pattern=pattern_str,
                    examples=info.get('examples', []),
                    frequency=info.get('frequency', 0),
                    confidence=info.get('confidence', 0.0),
                    last_seen=datetime.fromisoformat(info['last_seen']) if info.get('last_seen') else None
                )

            # Load stats
            stats = data.get('stats', {})
            if stats.get('last_update'):
                stats['last_update'] = datetime.fromisoformat(stats['last_update'])
            self._stats.update(stats)

            logger.info(f"Loaded {len(self.concepts)} concepts and {len(self.patterns)} patterns")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")

    @staticmethod
    def _update_vector(old_vector: np.ndarray, new_vector: np.ndarray,
                       learning_rate: float = 0.1) -> np.ndarray:
        return (1 - learning_rate) * old_vector + learning_rate * new_vector

    @staticmethod
    def _update_confidence(old_conf: float, new_conf: float,
                           weight: float = 0.1) -> float:
        return (1 - weight) * old_conf + weight * new_conf

    # یک خط خالی در اینجا


