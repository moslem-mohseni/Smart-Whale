from tokenization import Tokenizer
from normalization import Normalizer
from transliteration import Transliterator
from sentiment_analysis import SentimentAnalyzer
from grammar import GrammarProcessor
from semantics import SemanticProcessor

try:
    from ..teacher import TeacherModel
except ModuleNotFoundError:
    TeacherModel = None

from ..smart_model import SmartModel

# مقداردهی اولیه کلاس‌ها
Tokenizer = Tokenizer()
Normalizer = Normalizer()
Transliterator = Transliterator()
SentimentAnalyzer = SentimentAnalyzer()
GrammarProcessor = GrammarProcessor()
SemanticProcessor = SemanticProcessor()

__all__ = [
    "Tokenizer",
    "Normalizer",
    "Transliterator",
    "SentimentAnalyzer",
    "GrammarProcessor",
    "SemanticProcessor",
    "TeacherModel",
    "SmartModel",
]
