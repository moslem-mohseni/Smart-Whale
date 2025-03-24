import json
from docx import Document
from infrastructure.kafka.service.kafka_service import KafkaService


class WordCollector:
    """
    جمع‌آوری داده از فایل‌های Word و انتشار در Kafka
    """

    def __init__(self, kafka_topic):
        self.kafka_topic = kafka_topic
        self.kafka_service = KafkaService()

    def extract_text(self, file_path):
        """استخراج متن از فایل Word"""
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    def process_and_publish(self, file_path):
        """پردازش فایل Word و ارسال متن به Kafka"""
        text = self.extract_text(file_path)
        message = {"file": file_path, "text": text}
        self.kafka_service.send_message(self.kafka_topic, json.dumps(message, ensure_ascii=False))
        return len(text)


if __name__ == "__main__":
    kafka_topic = "word_data"
    collector = WordCollector(kafka_topic)
    file_path = "example.docx"

    try:
        text_length = collector.process_and_publish(file_path)
        print(f"✅ متن استخراج و {text_length} کاراکتر به Kafka ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در پردازش فایل Word: {e}")
