from .telegram_collector import TelegramScraper
from .twitter_collector import TwitterScraper
from .linkedin_collector import LinkedInScraper

# مقداردهی اولیه ماژول social
__all__ = ["TelegramScraper", "TwitterScraper", "LinkedInScraper"]