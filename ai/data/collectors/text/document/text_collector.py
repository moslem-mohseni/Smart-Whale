import json
from infrastructure.kafka.service.kafka_service import KafkaService


class TextCollector:
    """
    جمع‌آوری داده از فایل‌های متنی (`.txt`) و انتشار در Kafka
    """

    def __init__(self, kafka_topic):
        self.kafka_topic = kafka_topic
        self.kafka_service = KafkaService()

    def extract_text(self, file_path):
        """استخراج متن از فایل متنی"""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def process_and_publish(self, file_path):
        """پردازش فایل متنی و ارسال متن به Kafka"""
        text = self.extract_text(file_path)
        message = {"file": file_path, "text": text}
        self.kafka_service.send_message(self.kafka_topic, json.dumps(message, ensure_ascii=False))
        return len(text)


if __name__ == "__main__":
    kafka_topic = "text_data"
    collector = TextCollector(kafka_topic)
    file_path = "example.txt"

    try:
        text_length = collector.process_and_publish(file_path)
        print(f"✅ متن استخراج و {text_length} کاراکتر به Kafka ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در پردازش فایل متنی: {e}")