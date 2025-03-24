import asyncio
import logging
import aiohttp
from bs4 import BeautifulSoup
from collectors.base.collector import BaseCollector

logging.basicConfig(level=logging.INFO)


class GeneralWebCrawler(BaseCollector):
    """
    خزنده‌ی عمومی وب برای جمع‌آوری داده از صفحات HTML.
    """

    def __init__(self, source_name: str, start_url: str, max_pages: int = 5):
        super().__init__(source_name)
        self.start_url = start_url
        self.max_pages = max_pages
        self.visited_urls = set()

    async def collect_data(self):
        """
        اجرای فرآیند خزش و جمع‌آوری داده.
        """
        async with aiohttp.ClientSession() as session:
            return await self._crawl(session, self.start_url, 0)

    async def _crawl(self, session, url, depth):
        """
        تابع بازگشتی برای خزش در صفحات وب.
        """
        if depth >= self.max_pages or url in self.visited_urls:
            return None

        logging.info(f"🌍 Crawling: {url}")
        self.visited_urls.add(url)

        try:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                extracted_text = self._extract_text(soup)

                # استخراج لینک‌های جدید
                links = [a['href'] for a in soup.find_all('a', href=True)]

                # خزش لینک‌های جدید به صورت محدود
                for link in links[:2]:  # محدود کردن به 2 لینک برای جلوگیری از اضافه‌بار
                    if link.startswith("http"):
                        await self._crawl(session, link, depth + 1)

                return extracted_text
        except Exception as e:
            logging.error(f"❌ Error crawling {url}: {e}")
            return None

    def _extract_text(self, soup):
        """
        استخراج متن مفید از صفحه HTML.
        """
        return '\n'.join([p.get_text() for p in soup.find_all('p')])


if __name__ == "__main__":
    crawler = GeneralWebCrawler("TestCrawler", "https://example.com", max_pages=3)
    asyncio.run(crawler.start_collection())
