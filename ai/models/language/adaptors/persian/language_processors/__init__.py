from .contextual import ContextualKnowledge
from .dialects import PersianDialectProcessor
from .domain import DomainKnowledge
from .grammar import PersianGrammarProcessor
from .knowledge_store import KnowledgeGraph
from .literature import PersianLiteratureProcessor
from .proverbs import PersianProverbProcessor
from .semantics import PersianSemanticProcessor

# مقداردهی اولیه کلاس‌ها
contextual_knowledge = ContextualKnowledge()
dialect_processor = PersianDialectProcessor()
domain_knowledge = DomainKnowledge()
grammar_processor = PersianGrammarProcessor()
knowledge_graph = KnowledgeGraph()
literature_processor = PersianLiteratureProcessor()
proverb_processor = PersianProverbProcessor()
semantic_processor = PersianSemanticProcessor()

__all__ = [
    "contextual_knowledge",
    "dialect_processor",
    "domain_knowledge",
    "grammar_processor",
    "knowledge_graph",
    "literature_processor",
    "proverb_processor",
    "semantic_processor"
]