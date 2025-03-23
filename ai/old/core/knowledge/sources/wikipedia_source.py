import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import hashlib
from bs4 import BeautifulSoup

from ..interfaces.knowledge_source import (
    KnowledgeSource,
    KnowledgeSourceType,
    KnowledgeMetadata,
    KnowledgePriority,
    LearningContext
)

logger = logging.getLogger(__name__)


class WikipediaSource(KnowledgeSource):
    def __init__(self, source_config: Dict[str, Any]):
        super().__init__(source_config)
        self.session = None
        self.base_url = source_config.get('base_url', 'https://wikipedia.org/w/api.php')
        self.languages = source_config.get('languages', ['en', 'fa', 'ar'])
        self.max_retries = source_config.get('max_retries', 3)
        self.retry_delay = source_config.get('retry_delay', 1)
        self._cache = {}

    async def initialize(self) -> bool:
        try:
            self.session = aiohttp.ClientSession()
            headers = {'User-Agent': 'AI Learning Bot/1.0'}
            self.session.headers.update(headers)
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Wikipedia source: {str(e)}")
            return False

    async def _search_wikipedia(self, query: str, language: str) -> List[Dict]:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srwhat': 'text',
            'srlimit': 5,
        }

        try:
            async with self.session.get(
                    f"https://{language}.wikipedia.org/w/api.php",
                    params=params
            ) as response:
                data = await response.json()
                return data.get('query', {}).get('search', [])
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {str(e)}")
            return []

    async def _get_article_content(self, page_id: int, language: str) -> Optional[Dict]:
        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'extracts|categories|links',
            'pageids': page_id,
            'explaintext': True,
            'exsectionformat': 'plain'
        }

        try:
            async with self.session.get(
                    f"https://{language}.wikipedia.org/w/api.php",
                    params=params
            ) as response:
                data = await response.json()
                pages = data.get('query', {}).get('pages', {})
                return next(iter(pages.values()), None)
        except Exception as e:
            logger.error(f"Error fetching article content: {str(e)}")
            return None

    async def get_knowledge(self, query: str,
                            context: Optional[LearningContext] = None) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Wikipedia source is not initialized")

        cache_key = f"{query}_{context.language if context else 'en'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            language = context.language if context else 'en'
            search_results = await self._search_wikipedia(query, language)

            if not search_results:
                logger.warning(f"No results found for query: {query}")
                return self._create_empty_response()

            # دریافت محتوای اولین نتیجه
            article = await self._get_article_content(
                search_results[0]['pageid'],
                language
            )

            if not article:
                return self._create_empty_response()

            content = article.get('extract', '')
            categories = [cat['title'] for cat in article.get('categories', [])]

            metadata = KnowledgeMetadata(
                source_id=self.config['source_id'],
                source_type=KnowledgeSourceType.WIKIPEDIA,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                priority=self._determine_priority(categories),
                confidence_score=0.9,  # ویکی‌پدیا معمولاً منبع قابل اعتمادی است
                validation_status=True,
                learning_progress=0.0,
                tags=categories,
                language=language,
                size_bytes=len(content.encode('utf-8')),
                checksum=hashlib.sha256(content.encode()).hexdigest()
            )

            response = {
                'content': content,
                'metadata': metadata,
                'related_articles': search_results[1:],  # سایر نتایج مرتبط
                'categories': categories
            }

            self._cache[cache_key] = response
            return response

        except Exception as e:
            logger.error(f"Error in Wikipedia knowledge extraction: {str(e)}")
            raise

    def _determine_priority(self, categories: List[str]) -> KnowledgePriority:
        """تعیین اولویت بر اساس دسته‌بندی‌های مقاله"""
        important_keywords = ['science', 'technology', 'mathematics', 'علوم', 'فناوری', 'ریاضیات']

        for category in categories:
            if any(keyword in category.lower() for keyword in important_keywords):
                return KnowledgePriority.HIGH

        return KnowledgePriority.MEDIUM

    def _create_empty_response(self) -> Dict[str, Any]:
        """ایجاد پاسخ خالی برای زمانی که نتیجه‌ای یافت نشد"""
        return {
            'content': '',
            'metadata': None,
            'related_articles': [],
            'categories': []
        }

    async def validate_knowledge(self, knowledge_data: Dict[str, Any]) -> bool:
        """اعتبارسنجی دانش دریافت شده از ویکی‌پدیا"""
        try:
            content = knowledge_data.get('content', '')
            if not content:
                return False

            # بررسی حداقل طول محتوا
            if len(content) < 100:  # حداقل 100 کاراکتر
                return False

            # بررسی وجود منابع
            if 'related_articles' not in knowledge_data:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating Wikipedia knowledge: {str(e)}")
            return False

    async def update_learning_progress(self, knowledge_id: str, progress: float) -> None:
        """به‌روزرسانی پیشرفت یادگیری"""
        if not 0 <= progress <= 1:
            raise ValueError("Progress must be between 0 and 1")

        # فعلاً فقط لاگ می‌کنیم
        logger.info(f"Learning progress for Wikipedia article {knowledge_id}: {progress}")

    async def cleanup(self) -> None:
        """پاکسازی منابع"""
        if self.session:
            await self.session.close()
            self.session = None
        self._initialized = False
        self._cache.clear()
        logger.info("Wikipedia source cleaned up")