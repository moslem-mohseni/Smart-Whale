from .audio import VoiceCollector, MusicCollector, PodcastCollector
from .image import WebImageCollector, StorageImageCollector, SocialImageCollector
from .video import YouTubeVideoCollector, AparatVideoCollector, SocialVideoCollector

# مقداردهی اولیه ماژول text
__all__ = [
    "VoiceCollector", "MusicCollector", "PodcastCollector",
    "WebImageCollector", "StorageImageCollector", "SocialImageCollector",
    "YouTubeVideoCollector", "AparatVideoCollector", "SocialVideoCollector"
]
