import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)


class SourceManager:
    """
    مدیریت منابع داده برای جمع‌آوری داده از چندین منبع.
    این کلاس اطلاعات مربوط به منابع مختلف را مدیریت می‌کند.
    """

    def __init__(self):
        self.sources: Dict[str, Dict] = {}

    def add_source(self, source_name: str, config: Dict):
        """
        افزودن یک منبع جدید به سیستم.
        """
        if source_name in self.sources:
            logging.warning(f"⚠ Source {source_name} already exists.")
            return

        self.sources[source_name] = config
        logging.info(f"✅ Source {source_name} added successfully.")

    def remove_source(self, source_name: str):
        """
        حذف یک منبع از سیستم.
        """
        if source_name not in self.sources:
            logging.warning(f"⚠ Source {source_name} does not exist.")
            return

        del self.sources[source_name]
        logging.info(f"❌ Source {source_name} removed successfully.")

    def get_source_config(self, source_name: str) -> Dict:
        """
        دریافت پیکربندی یک منبع مشخص.
        """
        return self.sources.get(source_name, {})

    def list_sources(self):
        """
        لیست تمامی منابع فعال.
        """
        return list(self.sources.keys())


if __name__ == "__main__":
    manager = SourceManager()
    manager.add_source("WebCrawler", {"url": "https://example.com", "interval": 10})
    manager.add_source("TwitterAPI", {"api_key": "xyz", "interval": 5})
    print("Current Sources:", manager.list_sources())
    print("Twitter Config:", manager.get_source_config("TwitterAPI"))
    manager.remove_source("WebCrawler")
    print("Updated Sources:", manager.list_sources())