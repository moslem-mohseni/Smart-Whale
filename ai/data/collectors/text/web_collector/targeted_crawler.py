import asyncio
import logging
import aiohttp
from bs4 import BeautifulSoup
from collectors.base.collector import BaseCollector

logging.basicConfig(level=logging.INFO)


class TargetedWebCrawler(BaseCollector):
    """
    خزنده‌ی هدفمند وب برای جمع‌آوری داده از صفحات مشخص‌شده.
    """

    def __init__(self, source_name: str, target_urls: list):
        super().__init__(source_name)
        self.target_urls = target_urls

    async def collect_data(self):
        """
        اجرای فرآیند خزش هدفمند و جمع‌آوری داده.
        """
        async with aiohttp.ClientSession() as session:
            results = {}
            for url in self.target_urls:
                results[url] = await self._fetch_page(session, url)
            return results

    async def _fetch_page(self, session, url):
        """
        دریافت محتوای صفحه و استخراج داده.
        """
        logging.info(f"🎯 Targeted Crawling: {url}")
        try:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                return self._extract_text(soup)
        except Exception as e:
            logging.error(f"❌ Error fetching {url}: {e}")
            return None

    def _extract_text(self, soup):
        """
        استخراج متن مفید از صفحه HTML.
        """
        return '\n'.join([p.get_text() for p in soup.find_all('p')])


if __name__ == "__main__":
    crawler = TargetedWebCrawler("SpecificCrawler", ["https://example.com", "https://example.org"])
    asyncio.run(crawler.start_collection())
