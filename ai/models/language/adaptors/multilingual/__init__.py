from .language_processor import LanguageProcessor
from .language_processor.tokenization import Tokenizer
from .language_processor.normalization import Normalizer
from .language_processor.transliteration import Transliterator
from .language_processor.sentiment_analysis import SentimentAnalyzer
from .language_processor.grammar import GrammarProcessor
from .language_processor.semantics import SemanticProcessor
from .config import CONFIG

try:
    from .teacher import TeacherModel
except ModuleNotFoundError:
    TeacherModel = None

from .smart_model import SmartModel

__all__ = [
    "LanguageProcessor",
    "Tokenizer",
    "Normalizer",
    "Transliterator",
    "SentimentAnalyzer",
    "GrammarProcessor",
    "SemanticProcessor",
    "CONFIG",
    "TeacherModel",
    "SmartModel",
]
