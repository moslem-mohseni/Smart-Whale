from hazm import Normalizer, Lemmatizer, POSTagger, Chunker
from hazm import word_tokenize, sent_tokenize
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class HazmResult:
    original_text: str
    normalized_text: str
    sentences: List[str]
    words: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    pos_tags: List[Tuple[str, str]] = field(default_factory=list)
    chunks: List[Tuple[str, List[str]]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.words and self.normalized_text:
            self.words = word_tokenize(self.normalized_text)


class HazmAdapter:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.normalizer = Normalizer()
        self.lemmatizer = Lemmatizer()
        self.tagger = None
        self.chunker = None
        self._initialized = False
        logger.info("Creating Hazm adapter")

    async def initialize(self) -> bool:
        try:
            model_path = Path(self.config.get('hazm_resources_path', 'resources/hazm'))
            self.tagger = POSTagger(model=str(model_path / 'postagger.model'))
            self.chunker = Chunker(model=str(model_path / 'chunker.model'))
            self._initialized = True
            logger.info("Loaded Hazm models successfully")
            return True
        except Exception as e:
            self.tagger = POSTagger()
            self.chunker = Chunker()
            logger.warning(f"Using default Hazm models: {e}")
            self._initialized = True
            return True

    async def process_text(self, text: str) -> Optional[HazmResult]:
        try:
            normalized = self.normalizer.normalize(text)
            sentences = sent_tokenize(normalized)
            words = word_tokenize(normalized)

            if not words:
                return None

            lemmas = [self.lemmatizer.lemmatize(word) for word in words]
            pos_tags = self.tagger.tag(words)
            chunks = self.chunker.parse(pos_tags)

            return HazmResult(
                original_text=text,
                normalized_text=normalized,
                sentences=sentences,
                words=words,
                lemmas=lemmas,
                pos_tags=pos_tags,
                chunks=chunks
            )

        except Exception as e:
            logger.error(f"Error processing text with Hazm: {e}")
            return None

    async def extract_phrases(self, text: str) -> List[str]:
        result = await self.process_text(text)
        if not result or not result.pos_tags:
            return []

        phrases = []
        current_phrase = []
        for word, tag in result.pos_tags:
            if tag.startswith(('N', 'V')):
                current_phrase.append(word)
            elif current_phrase:
                phrases.append(' '.join(current_phrase))
                current_phrase = []

        if current_phrase:
            phrases.append(' '.join(current_phrase))

        return phrases

    def get_config(self) -> Dict[str, Any]:
        return {
            'is_initialized': self._initialized,
            'model_path': self.config.get('hazm_resources_path'),
            'models_loaded': {
                'tagger': self.tagger is not None,
                'chunker': self.chunker is not None
            }
        }