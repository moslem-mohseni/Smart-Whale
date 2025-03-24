import torch
import os
from ai.models.language.infrastructure.redis_connector import RedisConnector
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class ModelUpdater:
    """
    مدیریت بروزرسانی مدل‌های زبانی، ذخیره نسخه‌ها، مقایسه‌ی عملکرد مدل‌ها، و کنترل کیفیت قبل از جایگزینی نسخه‌ی قبلی.
    """

    def __init__(self, model, model_dir="ai/models/language/trained_models/"):
        self.model = model
        self.model_dir = model_dir

        # استفاده از سرویس‌های infrastructure
        self.redis_client = RedisConnector()
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای مدیریت نسخه‌های مدل در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره‌ی اطلاعات مربوط به نسخه‌های مختلف مدل‌های زبانی.
        """
        query = """
        CREATE TABLE IF NOT EXISTS model_versions (
            model_name String,
            version String,
            accuracy Float32,
            loss Float32,
            deployment_status String,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def save_model(self, version):
        """
        ذخیره‌ی نسخه‌ی جدید مدل در سیستم فایل.
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_path = os.path.join(self.model_dir, f"{self.model.__class__.__name__}_v{version}.pt")
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def load_model(self, version):
        """
        بارگذاری نسخه‌ی مشخصی از مدل از سیستم فایل.
        """
        model_path = os.path.join(self.model_dir, f"{self.model.__class__.__name__}_v{version}.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            return True
        return False

    def get_latest_version(self, model_name):
        """
        دریافت آخرین نسخه‌ی مدل از Redis یا ClickHouse.
        """
        redis_key = f"model_version:{model_name}"
        latest_version = self.redis_client.get(redis_key)

        if latest_version:
            return latest_version

        query = f"""
        SELECT version FROM model_versions
        WHERE model_name = '{model_name}'
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            latest_version = result[0][0]
            self.redis_client.set(redis_key, latest_version, ex=86400)  # کش نسخه‌ی مدل برای 24 ساعت
            return latest_version

        return None

    def update_model(self, new_version, accuracy, loss):
        """
        بروزرسانی مدل در صورتی که عملکرد نسخه‌ی جدید بهتر از نسخه‌ی قبلی باشد.
        """
        latest_version = self.get_latest_version(self.model.__class__.__name__)

        if latest_version:
            # دریافت عملکرد نسخه‌ی قبلی از ClickHouse
            query = f"""
            SELECT accuracy, loss FROM model_versions
            WHERE model_name = '{self.model.__class__.__name__}' AND version = '{latest_version}'
            LIMIT 1;
            """
            result = self.clickhouse_client.execute_query(query)

            if result:
                prev_accuracy, prev_loss = result[0]

                # بررسی اینکه آیا نسخه‌ی جدید عملکرد بهتری دارد؟
                if accuracy < prev_accuracy and loss > prev_loss:
                    print(f"🚨 نسخه‌ی جدید مدل عملکرد بدتری دارد، بروزرسانی لغو شد.")
                    return False

        # ذخیره‌ی نسخه‌ی جدید مدل
        model_path = self.save_model(new_version)

        # ثبت اطلاعات نسخه‌ی جدید در ClickHouse
        query = f"""
        INSERT INTO model_versions (model_name, version, accuracy, loss, deployment_status)
        VALUES ('{self.model.__class__.__name__}', '{new_version}', {accuracy}, {loss}, 'DEPLOYED');
        """
        self.clickhouse_client.execute_query(query)

        # بروزرسانی نسخه‌ی مدل در Redis
        redis_key = f"model_version:{self.model.__class__.__name__}"
        self.redis_client.set(redis_key, new_version, ex=86400)

        print(f"✅ مدل {self.model.__class__.__name__} به نسخه‌ی {new_version} بروزرسانی شد و آماده‌ی استفاده است.")
        return True

# تست عملکرد
if __name__ == "__main__":
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel()
    updater = ModelUpdater(model)

    # ذخیره و بروزرسانی مدل در صورت داشتن عملکرد بهتر
    updater.update_model(new_version="1.1", accuracy=0.92, loss=0.05)
