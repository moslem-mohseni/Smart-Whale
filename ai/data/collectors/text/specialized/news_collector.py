import requests
import json
from infrastructure.kafka.service.kafka_service import KafkaService


class NewsCollector:
    """
    کلاس جمع‌آوری اخبار از منابع مختلف خبری و انتشار روی Kafka
    """

    def __init__(self, api_key, source_url, kafka_topic, max_articles=20):
        self.api_key = api_key
        self.source_url = source_url
        self.kafka_topic = kafka_topic
        self.max_articles = max_articles
        self.kafka_service = KafkaService()

    def fetch_news(self):
        """دریافت اخبار از API و انتشار روی Kafka"""
        response = requests.get(self.source_url, headers={"Authorization": f"Bearer {self.api_key}"})
        if response.status_code != 200:
            raise ValueError(f"⚠ خطا در دریافت اخبار: {response.status_code}")

        news_data = response.json()
        articles = news_data.get("articles", [])[:self.max_articles]

        for article in articles:
            self.kafka_service.send_message(self.kafka_topic, json.dumps(article, ensure_ascii=False))

        return len(articles)


if __name__ == "__main__":
    api_key = "your_api_key_here"
    source_url = "https://newsapi.org/v2/top-headlines?country=us"
    kafka_topic = "news_data"

    collector = NewsCollector(api_key, source_url, kafka_topic, max_articles=10)
    try:
        news_count = collector.fetch_news()
        print(f"✅ {news_count} مقاله خبری دریافت و به Kafka ارسال شد.")
    except ValueError as e:
        print(e)
