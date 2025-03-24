import schedule
import time
import threading
from ai.models.language.learning.trainer.learning_pipeline import LearningPipeline
from ai.models.language.infrastructure.kafka.kafka_producer import KafkaProducer
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class LearningScheduler:
    """
    مدیریت زمان‌بندی فرآیند یادگیری و اجرای خودکار مدل‌های آموزشی.
    """

    def __init__(self, teacher_model, student_model, schedule_interval="daily"):
        self.pipeline = LearningPipeline(teacher_model, student_model)
        self.kafka_producer = KafkaProducer()
        self.clickhouse_client = ClickHouseDB()
        self.schedule_interval = schedule_interval

        # ایجاد جدول برای ثبت فرآیندهای زمان‌بندی‌شده در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره اطلاعات فرآیندهای یادگیری زمان‌بندی‌شده.
        """
        query = """
        CREATE TABLE IF NOT EXISTS learning_schedule_log (
            model_name String,
            execution_time DateTime DEFAULT now(),
            status String,
            message String
        ) ENGINE = MergeTree()
        ORDER BY (model_name, execution_time);
        """
        self.clickhouse_client.execute_query(query)

    def log_schedule_execution(self, model_name, status, message):
        """
        ثبت فرآیندهای زمان‌بندی‌شده در ClickHouse.
        """
        query = f"""
        INSERT INTO learning_schedule_log (model_name, status, message)
        VALUES ('{model_name}', '{status}', '{message}');
        """
        self.clickhouse_client.execute_query(query)

    def execute_scheduled_learning(self):
        """
        اجرای فرآیند یادگیری و ثبت وضعیت در ClickHouse.
        """
        try:
            print("📌 فرآیند یادگیری زمان‌بندی‌شده در حال اجرا است...")
            raw_data = ["این یک متن تستی است.", "مثال دیگر از داده‌های پردازشی."]

            # داده‌های ساختگی برای تست یادگیری
            import torch
            from torch.utils.data import DataLoader, TensorDataset

            train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
            val_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))

            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

            # اجرای یادگیری
            self.pipeline.execute_pipeline(raw_data, train_loader, val_loader, version="1.2")

            # ارسال پیام به Kafka درباره موفقیت فرآیند یادگیری
            self.kafka_producer.send_message("learning_schedule", {"status": "success", "message": "Learning executed successfully."})

            # ثبت موفقیت فرآیند در ClickHouse
            self.log_schedule_execution(self.pipeline.student_model.__class__.__name__, "Success", "Learning executed successfully.")
        except Exception as e:
            print(f"🚨 خطا در فرآیند یادگیری: {str(e)}")
            self.log_schedule_execution(self.pipeline.student_model.__class__.__name__, "Failed", str(e))

    def start_scheduler(self):
        """
        تنظیم زمان‌بندی اجرای فرآیند یادگیری و اجرای خودکار آن.
        """
        if self.schedule_interval == "daily":
            schedule.every().day.at("02:00").do(self.execute_scheduled_learning)
        elif self.schedule_interval == "weekly":
            schedule.every().monday.at("03:00").do(self.execute_scheduled_learning)
        elif self.schedule_interval == "hourly":
            schedule.every().hour.do(self.execute_scheduled_learning)
        else:
            print("🚨 مقدار نامعتبر برای زمان‌بندی!")

        print(f"✅ فرآیند یادگیری زمان‌بندی‌شده تنظیم شد: {self.schedule_interval}")

        while True:
            schedule.run_pending()
            time.sleep(60)

# تست عملکرد
if __name__ == "__main__":
    import torch.nn as nn
    import threading

    class TeacherModel(nn.Module):
        def __init__(self):
            super(TeacherModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    class StudentModel(nn.Module):
        def __init__(self):
            super(StudentModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    teacher_model = TeacherModel()
    student_model = StudentModel()

    scheduler = LearningScheduler(teacher_model, student_model, schedule_interval="daily")

    # اجرای زمان‌بندی در یک ترد جداگانه برای جلوگیری از بلاک شدن برنامه اصلی
    scheduler_thread = threading.Thread(target=scheduler.start_scheduler)
    scheduler_thread.start()
