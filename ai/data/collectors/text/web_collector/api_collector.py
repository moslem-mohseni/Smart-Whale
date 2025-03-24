import asyncio
import logging
import aiohttp
from collectors.base.collector import BaseCollector

logging.basicConfig(level=logging.INFO)


class APICollector(BaseCollector):
    """
    جمع‌آوری داده از APIهای وب با استفاده از درخواست‌های HTTP.
    """

    def __init__(self, source_name: str, api_url: str, headers: dict = None, params: dict = None):
        super().__init__(source_name)
        self.api_url = api_url
        self.headers = headers or {}
        self.params = params or {}

    async def collect_data(self):
        """
        ارسال درخواست به API و دریافت داده.
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.api_url, headers=self.headers, params=self.params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logging.error(f"❌ API request failed: {response.status}")
                        return None
            except Exception as e:
                logging.error(f"❌ Error fetching API data: {e}")
                return None


if __name__ == "__main__":
    collector = APICollector("TestAPI", "https://jsonplaceholder.typicode.com/posts")
    asyncio.run(collector.start_collection())
