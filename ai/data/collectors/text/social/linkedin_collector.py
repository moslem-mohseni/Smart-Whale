import time
import logging
import pickle
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from infrastructure.redis.service.cache_service import CacheService
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.clickhouse.service.analytics_service import AnalyticsService


# تنظیمات لاگ
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("linkedin_scraper_debug.log", encoding="utf-8"), logging.StreamHandler()]
    )


setup_logging()

# مسیر ذخیره‌سازی سشن‌ها
COOKIES_FILE = "linkedin_cookies.pkl"
CHROME_PROFILE_PATH = r"C:\Users\ASUS\AppData\Local\Google\Chrome\User Data\Profile 7"

# سرویس‌های `Redis`، `Kafka` و `ClickHouse`
cache_service = CacheService()
kafka_service = KafkaService()
analytics_service = AnalyticsService()


class LinkedInScraper:
    def __init__(self, headless=False, max_posts=20):
        self.max_posts = max_posts
        self.posts = set()
        self.driver = self._setup_driver(headless)
        self._load_cookies()

    def _setup_driver(self, headless):
        chrome_options = Options()
        chrome_options.add_argument(f"--user-data-dir={CHROME_PROFILE_PATH}")
        chrome_options.add_argument("--profile-directory=Default")
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        try:
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            logging.info("✅ WebDriver مقداردهی شد.")
            return driver
        except Exception as e:
            logging.error(f"❌ خطا در مقداردهی WebDriver: {e}")
            return None

    def _load_cookies(self):
        if os.path.exists(COOKIES_FILE):
            with open(COOKIES_FILE, "rb") as f:
                cookies = pickle.load(f)
                for cookie in cookies:
                    self.driver.add_cookie(cookie)
            logging.info("✅ کوکی‌ها بارگذاری شدند.")
        else:
            self._login_and_save_cookies()

    def _login_and_save_cookies(self):
        logging.info("🔹 کوکی‌ها موجود نیستند. نیاز به ورود به حساب کاربری دارید.")
        self.driver.get("https://www.linkedin.com/feed/")
        input("🔹 لطفاً وارد حساب کاربری شوید و سپس Enter را بزنید...")
        pickle.dump(self.driver.get_cookies(), open(COOKIES_FILE, "wb"))
        logging.info("✅ کوکی‌ها ذخیره شدند!")

    def search_posts(self, query):
        if not self.driver:
            logging.error("❌ WebDriver مقداردهی نشده است.")
            return []

        search_url = f"https://www.linkedin.com/search/results/content/?keywords={query}"
        logging.info(f"🔍 باز کردن جستجو: {search_url}")
        self.driver.get(search_url)
        time.sleep(5)

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        post_elements = soup.find_all("div", {"class": "feed-shared-update-v2"})

        for post in post_elements[:self.max_posts]:
            text = post.get_text().strip()
            if text and text not in self.posts:
                self.posts.add(text)
                cache_service.set("linkedin_posts", list(self.posts))
                kafka_service.send_message("linkedin_posts", text)
                analytics_service.insert_row("linkedin_data", {"text": text, "created_at": "NOW()"})
                logging.info(f"✅ پست جدید ذخیره و پردازش شد: {text[:50]}...")

        return list(self.posts)

    def close(self):
        if self.driver:
            self.driver.quit()
            logging.info("✅ WebDriver بسته شد.")


if __name__ == "__main__":
    scraper = LinkedInScraper(headless=False, max_posts=10)
    posts = scraper.search_posts("هوش مصنوعی")

    if posts:
        logging.info(f"✅ {len(posts)} پست جمع‌آوری شد.")
    else:
        logging.warning("⚠️ هیچ پستی دریافت نشد.")

    scraper.close()