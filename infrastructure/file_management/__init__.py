import logging

# Importing all submodules to ensure proper initialization
from .adapters import *
from .cache import *
from .config import *
from .domain import *
from .monitoring import *
from .security import *
from .service import *
from .storage import *

logger = logging.getLogger(__name__)
logger.info("Initializing File Management Module...")

__all__ = [
    "adapters",
    "cache",
    "config",
    "domain",
    "monitoring",
    "security",
    "service",
    "storage"
]
