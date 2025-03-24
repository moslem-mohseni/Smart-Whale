import logging
from .deduplication import Deduplication
from .file_store import FileStore
from .lifecycle import FileLifecycle

logger = logging.getLogger(__name__)
logger.info("Initializing Storage Module...")

__all__ = [
    "Deduplication",
    "FileStore",
    "FileLifecycle"
]
