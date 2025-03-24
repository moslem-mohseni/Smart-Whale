from .document import PDFCollector, WordCollector, TextCollector
from .social import TelegramScraper, TwitterScraper, LinkedInScraper
from .specialized import WikiCollector, BooksScraper, NewsCollector
from .web_collector import GeneralWebCrawler, APICollector, TargetedWebCrawler

# مقداردهی اولیه ماژول text
__all__ = [
    "PDFCollector", "WordCollector", "TextCollector",
    "TelegramScraper", "TwitterScraper", "LinkedInScraper",
    "WikiCollector", "BooksScraper", "NewsCollector",
    "GeneralWebCrawler", "APICollector", "TargetedWebCrawler"
]