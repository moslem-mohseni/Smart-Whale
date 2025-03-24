from .base import BaseCollector, SourceManager, ErrorHandler
from .text import (
    PDFCollector, WordCollector, TextCollector,
    TelegramScraper, TwitterScraper, LinkedInScraper,
    WikiCollector, BooksScraper, NewsCollector,
    GeneralWebCrawler, APICollector, TargetedWebCrawler
)
from .media import (
    VoiceCollector, MusicCollector, PodcastCollector,
    WebImageCollector, StorageImageCollector, SocialImageCollector,
    YouTubeVideoCollector, AparatVideoCollector, SocialVideoCollector
)
from .composite import HybridCollector, RelationManager

# مقداردهی اولیه ماژول Collectors
source_manager = SourceManager()
error_handler = ErrorHandler()

__all__ = [
    "BaseCollector", "source_manager", "error_handler",
    "PDFCollector", "WordCollector", "TextCollector",
    "TelegramScraper", "TwitterScraper", "LinkedInScraper",
    "WikiCollector", "BooksScraper", "NewsCollector",
    "GeneralWebCrawler", "APICollector", "TargetedWebCrawler",
    "VoiceCollector", "MusicCollector", "PodcastCollector",
    "WebImageCollector", "StorageImageCollector", "SocialImageCollector",
    "YouTubeVideoCollector", "AparatVideoCollector", "SocialVideoCollector",
    "HybridCollector", "RelationManager"
]