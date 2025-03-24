import time
from ai.models.language.learning.trainer.clickhouse_logger import ClickHouseLogger
from ai.models.language.infrastructure.kafka.kafka_producer import KafkaProducer

class TrainingMonitor:
    """
    پایش و نظارت بر عملکرد فرآیند یادگیری مدل‌ها در طول اجرا.
    """

    def __init__(self):
        self.logger = ClickHouseLogger()
        self.kafka_producer = KafkaProducer()

    def monitor_training(self, model_name, version, threshold_loss=0.1, threshold_accuracy=0.85, check_interval=10):
        """
        بررسی روند یادگیری مدل و هشدار در صورت افت عملکرد.
        """
        while True:
            print(f"📊 در حال بررسی متریک‌های آموزشی مدل {model_name} - نسخه {version}...")

            # دریافت آخرین وضعیت یادگیری از ClickHouse
            history = self.logger.get_training_history(model_name, version, limit=1)

            if not history:
                print("🚨 هیچ داده‌ای برای نظارت یافت نشد!")
                time.sleep(check_interval)
                continue

            epoch, loss, accuracy, _, _, timestamp = history[0]

            print(f"🔹 Epoch {epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            # بررسی افت عملکرد مدل
            if loss > threshold_loss:
                message = f"🚨 هشدار: افزایش Loss در مدل {model_name} - نسخه {version} (Loss = {loss:.4f})"
                print(message)
                self.kafka_producer.send_message("training_alerts", {"status": "warning", "message": message})

            if accuracy < threshold_accuracy:
                message = f"🚨 هشدار: کاهش دقت مدل {model_name} - نسخه {version} (Accuracy = {accuracy:.4f})"
                print(message)
                self.kafka_producer.send_message("training_alerts", {"status": "warning", "message": message})

            time.sleep(check_interval)  # بررسی وضعیت یادگیری در فواصل زمانی مشخص

# تست عملکرد
if __name__ == "__main__":
    monitor = TrainingMonitor()

    # اجرای نظارت بر یک مدل فرضی
    monitor.monitor_training(model_name="TestModel", version="1.2", threshold_loss=0.2, threshold_accuracy=0.80, check_interval=15)
