from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class ClickHouseLogger:
    """
    مدیریت ثبت متریک‌های یادگیری مدل‌ها در ClickHouse برای تحلیل و بهینه‌سازی فرآیند آموزش.
    """

    def __init__(self):
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ذخیره متریک‌های آموزشی در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره اطلاعات آموزش مدل.
        """
        query = """
        CREATE TABLE IF NOT EXISTS training_metrics (
            model_name String,
            version String,
            epoch Int32,
            loss Float32,
            accuracy Float32,
            batch_size Int32,
            learning_rate Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def log_training_metrics(self, model_name, version, epoch, loss, accuracy, batch_size, learning_rate):
        """
        ثبت متریک‌های آموزشی در ClickHouse.
        """
        query = f"""
        INSERT INTO training_metrics (model_name, version, epoch, loss, accuracy, batch_size, learning_rate)
        VALUES ('{model_name}', '{version}', {epoch}, {loss}, {accuracy}, {batch_size}, {learning_rate});
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ متریک‌های آموزشی مدل {model_name} - نسخه {version} در ClickHouse ثبت شد.")

    def get_training_history(self, model_name, version, limit=10):
        """
        دریافت تاریخچه‌ی متریک‌های آموزشی برای یک مدل خاص.
        """
        query = f"""
        SELECT epoch, loss, accuracy, batch_size, learning_rate, timestamp
        FROM training_metrics
        WHERE model_name = '{model_name}' AND version = '{version}'
        ORDER BY timestamp DESC
        LIMIT {limit};
        """
        result = self.clickhouse_client.execute_query(query)
        return result

# تست عملکرد
if __name__ == "__main__":
    logger = ClickHouseLogger()

    # ثبت داده تستی
    logger.log_training_metrics(
        model_name="TestModel",
        version="1.0",
        epoch=5,
        loss=0.08,
        accuracy=0.92,
        batch_size=32,
        learning_rate=0.001
    )

    # دریافت تاریخچه آموزشی
    history = logger.get_training_history("TestModel", "1.0")
    print("📊 تاریخچه‌ی آموزشی:", history)
