import pymupdf as fitz
import json
from infrastructure.kafka.service.kafka_service import KafkaService


class PDFCollector:
    """
    جمع‌آوری داده از فایل‌های PDF و انتشار در Kafka
    """

    def __init__(self, kafka_topic):
        self.kafka_topic = kafka_topic
        self.kafka_service = KafkaService()

    def extract_text(self, file_path):
        """استخراج متن از فایل PDF"""
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text

    def process_and_publish(self, file_path):
        """پردازش فایل PDF و ارسال متن به Kafka"""
        text = self.extract_text(file_path)
        message = {"file": file_path, "text": text}
        self.kafka_service.send_message(self.kafka_topic, json.dumps(message, ensure_ascii=False))
        return len(text)


if __name__ == "__main__":
    kafka_topic = "pdf_data"
    collector = PDFCollector(kafka_topic)
    file_path = "example.pdf"

    try:
        text_length = collector.process_and_publish(file_path)
        print(f"✅ متن استخراج و {text_length} کاراکتر به Kafka ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در پردازش فایل PDF: {e}")
