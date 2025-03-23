import time
import random
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from infrastructure.redis.service.cache_service import CacheService
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.kafka.service.topic_manager import TopicManager
from infrastructure.clickhouse.service.analytics_service import AnalyticsService

# تنظیم پروفایل کروم
CHROME_PROFILE_PATH = r"C:\Users\ASUS\AppData\Local\Google\Chrome\User Data\Profile 7"

# سرویس‌های `Redis`، `Kafka` و `ClickHouse`
cache_service = CacheService()
kafka_service = KafkaService()
topic_manager = TopicManager()
analytics_service = AnalyticsService()

# ایجاد `Topic` در `Kafka` در صورت نیاز
topic_manager.create_topic("tweets_topic")

# تنظیمات لاگ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("twitter_scraper_debug.log", encoding="utf-8"), logging.StreamHandler()]
)


class TwitterScraper:
    def __init__(self, headless=False, max_tweets=50):
        self.max_tweets = max_tweets
        self.tweets = set()
        self.load_cached_tweets()

        # تنظیم WebDriver
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
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            logging.info("✅ WebDriver مقداردهی شد.")
        except Exception as e:
            logging.error(f"❌ خطا در مقداردهی WebDriver: {e}")
            self.driver = None

    def load_cached_tweets(self):
        """بارگذاری توییت‌های کش شده از `Redis`"""
        cached_tweets = cache_service.get("tweets") or []
        self.tweets = set(cached_tweets)
        logging.info(f"🔹 {len(self.tweets)} توییت از `Redis` بارگذاری شد.")

    def search_tweets(self, query):
        """جستجو و استخراج توییت‌ها"""
        if not self.driver:
            logging.error("❌ WebDriver مقداردهی نشده است.")
            return []

        search_url = f"https://x.com/search?q={query}&src=typed_query&f=live"
        logging.info(f"🔍 باز کردن جستجو: {search_url}")
        self.driver.get(search_url)
        time.sleep(5)

        last_height = self.driver.execute_script("return document.body.scrollHeight")
        iteration = 0

        while len(self.tweets) < self.max_tweets:
            iteration += 1
            logging.info(f"🔄 پردازش صفحه {iteration}")

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            tweet_elements = soup.find_all("div", {"data-testid": "tweetText"})
            logging.info(f"🔍 تعداد توییت‌های یافت شده در صفحه {iteration}: {len(tweet_elements)}")

            for tweet in tweet_elements:
                text = tweet.get_text().strip()
                if text and text not in self.tweets:
                    self.tweets.add(text)
                    cache_service.set("tweets", list(self.tweets))  # ذخیره در `Redis`
                    kafka_service.send_message("tweets_topic", text)  # ارسال به `Kafka`
                    analytics_service.insert_row("tweets", {"text": text, "created_at": "NOW()"})  # ذخیره در `ClickHouse`
                    logging.info(f"✅ توییت جدید ذخیره و پردازش شد: {text[:50]}...")

                if len(self.tweets) >= self.max_tweets:
                    logging.info("🎯 تعداد موردنظر توییت‌ها جمع‌آوری شد.")
                    break

            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(2, 5))

            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                logging.info("🔚 به انتهای صفحه رسیدیم.")
                break
            last_height = new_height

        return list(self.tweets)

    def close(self):
        """بستن WebDriver."""
        if self.driver:
            self.driver.quit()
            logging.info("✅ WebDriver بسته شد.")


# =========================
# ✅ **اجرای اسکرپر**
# =========================
if __name__ == "__main__":
    scraper = TwitterScraper(headless=False, max_tweets=20)
    tweets = scraper.search_tweets("هوش مصنوعی")

    if tweets:
        logging.info(f"✅ {len(tweets)} توییت جمع‌آوری شد.")
    else:
        logging.warning("⚠️ هیچ توییتی دریافت نشد.")

    scraper.close()
