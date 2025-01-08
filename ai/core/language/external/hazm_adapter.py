from hazm import Normalizer, Lemmatizer, POSTagger, Chunker
from hazm import word_tokenize, sent_tokenize
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class HazmResult:
    original_text: str
    normalized_text: str
    sentences: List[str]
    words: List[str]
    lemmas: List[str]
    pos_tags: List[Tuple[str, str]]
    chunks: List[Tuple[str, List[str]]]
    timestamp: datetime

class HazmAdapter:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.normalizer = None
        self.lemmatizer = None
        self.tagger = None
        self.chunker = None
        self._initialized = False
        logger.info("Creating Hazm adapter")

    async def initialize(self) -> bool:
        try:
            model_path = Path(self.config.get('hazm_resources_path', 'resources/hazm'))
            if not model_path.exists():
                model_path.mkdir(parents=True, exist_ok=True)

            self.normalizer = Normalizer()
            self.lemmatizer = Lemmatizer()

            tagger_path = model_path / 'postagger.model'
            chunker_path = model_path / 'chunker.model'

            if not tagger_path.exists() or not chunker_path.exists():
                logger.warning(f"Models not found at {model_path}. Using default models.")
                self.tagger = POSTagger()
                self.chunker = Chunker()
            else:
                self.tagger = POSTagger(model=str(tagger_path))
                self.chunker = Chunker(model=str(chunker_path))

            self._initialized = True
            logger.info("Hazm tools initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Hazm tools: {e}")
            return False

    async def process_text(self, text: str) -> Optional[HazmResult]:
        try:
            if not self._initialized:
                await self.initialize()

            normalized = self.normalizer.normalize(text)
            sentences = sent_tokenize(normalized)
            words = word_tokenize(normalized)
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
                chunks=chunks,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error processing text with Hazm: {e}")
            return None

    async def extract_phrases(self, text: str) -> List[str]:
        try:
            result = await self.process_text(text)
            if not result:
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
        except Exception as e:
            logger.error(f"Error extracting phrases: {e}")
            return []

    def get_config(self) -> Dict[str, Any]:
        return {
            'is_initialized': self._initialized,
            'normalizer_available': self.normalizer is not None,
            'lemmatizer_available': self.lemmatizer is not None,
            'tagger_available': self.tagger is not None,
            'chunker_available': self.chunker is not None,
            'model_path': self.config.get('hazm_resources_path')
        }