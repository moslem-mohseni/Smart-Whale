import torch
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class BiasDetector:
    """
    بررسی میزان سوگیری مدل‌های زبانی و ثبت متریک‌های تحلیل Bias در ClickHouse.
    """

    def __init__(self):
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ذخیره اطلاعات تحلیل سوگیری مدل در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره متریک‌های سوگیری مدل‌های زبانی.
        """
        query = """
        CREATE TABLE IF NOT EXISTS model_bias (
            model_name String,
            version String,
            bias_category String,
            bias_score Float32,
            dataset_group String,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def calculate_bias(self, model, dataset):
        """
        محاسبه میزان سوگیری مدل نسبت به گروه‌های مختلف داده.
        """
        model.eval()
        group_accuracies = {}

        with torch.no_grad():
            for group, data_loader in dataset.items():
                correct = 0
                total = 0

                for inputs, labels in data_loader:
                    outputs = model(inputs)
                    predicted = torch.argmax(outputs, dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                accuracy = correct / total if total > 0 else 0
                group_accuracies[group] = accuracy

        return group_accuracies

    def log_bias_metrics(self, model_name, version, bias_category, bias_score, dataset_group):
        """
        ثبت میزان سوگیری مدل در ClickHouse.
        """
        query = f"""
        INSERT INTO model_bias (model_name, version, bias_category, bias_score, dataset_group)
        VALUES ('{model_name}', '{version}', '{bias_category}', {bias_score}, '{dataset_group}');
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ متریک‌های سوگیری مدل {model_name} - نسخه {version} در ClickHouse ثبت شد.")

    def get_bias_history(self, model_name, version, limit=10):
        """
        دریافت تاریخچه‌ی سوگیری مدل در ClickHouse.
        """
        query = f"""
        SELECT bias_category, bias_score, dataset_group, timestamp
        FROM model_bias
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
    bias_detector = BiasDetector()

    # داده‌های ساختگی برای تست تحلیل Bias
    dataset = {
        "Male": DataLoader(TensorDataset(torch.randn(50, 10), torch.randint(0, 2, (50,))), batch_size=16, shuffle=False),
        "Female": DataLoader(TensorDataset(torch.randn(50, 10), torch.randint(0, 2, (50,))), batch_size=16, shuffle=False),
        "Other": DataLoader(TensorDataset(torch.randn(50, 10), torch.randint(0, 2, (50,))), batch_size=16, shuffle=False),
    }

    # محاسبه سوگیری مدل
    bias_scores = bias_detector.calculate_bias(model, dataset)

    # ثبت سوگیری مدل در ClickHouse
    for group, score in bias_scores.items():
        bias_detector.log_bias_metrics("TestModel", "1.0", "Gender Bias", score, group)

    # دریافت تاریخچه سوگیری
    history = bias_detector.get_bias_history("TestModel", "1.0")
    print("📊 تاریخچه‌ی سوگیری مدل:", history)
