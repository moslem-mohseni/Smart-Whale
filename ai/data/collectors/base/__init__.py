from .collector import BaseCollector
from .source_manager import SourceManager
from .error_handler import ErrorHandler

# مقداردهی اولیه ماژول Collectors
source_manager = SourceManager()
error_handler = ErrorHandler()

__all__ = ["BaseCollector", "source_manager", "error_handler"]