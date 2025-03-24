import torch
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class AccuracyChecker:
    """
    بررسی دقت مدل‌های زبانی در طول فرآیند یادگیری و ثبت نتایج در ClickHouse.
    """

    def __init__(self):
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ذخیره دقت مدل‌ها در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره اطلاعات دقت مدل‌های زبانی.
        """
        query = """
        CREATE TABLE IF NOT EXISTS model_accuracy (
            model_name String,
            version String,
            epoch Int32,
            accuracy Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def calculate_accuracy(self, model, data_loader):
        """
        محاسبه دقت مدل بر روی داده‌های آزمایشی.
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def log_accuracy(self, model_name, version, epoch, accuracy):
        """
        ثبت دقت مدل در ClickHouse برای تحلیل‌های بعدی.
        """
        query = f"""
        INSERT INTO model_accuracy (model_name, version, epoch, accuracy)
        VALUES ('{model_name}', '{version}', {epoch}, {accuracy});
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ دقت مدل {model_name} - نسخه {version} برای Epoch {epoch} ثبت شد: {accuracy:.4f}")

    def get_accuracy_history(self, model_name, version, limit=10):
        """
        دریافت تاریخچه‌ی دقت مدل در ClickHouse.
        """
        query = f"""
        SELECT epoch, accuracy, timestamp
        FROM model_accuracy
        WHERE model_name = '{model_name}' AND version = '{version}'
        ORDER BY timestamp DESC
        LIMIT {limit};
        """
        result = self.clickhouse_client.execute_query(query)
        return result

# تست عملکرد
if __name__ == "__main__":
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel()
    accuracy_checker = AccuracyChecker()

    # داده‌های ساختگی برای تست
    test_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # محاسبه دقت مدل
    accuracy = accuracy_checker.calculate_accuracy(model, test_loader)

    # ثبت دقت مدل در ClickHouse
    accuracy_checker.log_accuracy("TestModel", "1.0", epoch=5, accuracy=accuracy)

    # دریافت تاریخچه دقت
    history = accuracy_checker.get_accuracy_history("TestModel", "1.0")
    print("📊 تاریخچه دقت مدل:", history)
