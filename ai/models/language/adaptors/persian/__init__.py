from .config import CONFIG
from .language_processor import PersianLanguageProcessor
from .smart_model import SmartModel
from .teacher import TeacherModel
from .language_processors import *

# مقداردهی اولیه
language_processor = PersianLanguageProcessor()
smart_model = SmartModel()
teacher_model = TeacherModel()

__all__ = [
    "CONFIG",
    "PersianLanguageProcessor",
    "SmartModel",
    "TeacherModel",
    "language_processor",
    "smart_model",
    "teacher_model",
    "contextual_knowledge",
    "dialect_processor",
    "domain_knowledge",
    "grammar_processor",
    "knowledge_graph",
    "literature_processor",
    "proverb_processor",
    "semantic_processor"
]