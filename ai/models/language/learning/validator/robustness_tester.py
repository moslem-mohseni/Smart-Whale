import torch
import numpy as np
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class RobustnessTester:
    """
    ارزیابی استحکام مدل در برابر تغییرات کوچک در داده‌های ورودی و ثبت نتایج در ClickHouse.
    """

    def __init__(self):
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ذخیره اطلاعات تست استحکام مدل در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره نتایج تست استحکام مدل.
        """
        query = """
        CREATE TABLE IF NOT EXISTS model_robustness (
            model_name String,
            version String,
            perturbation_type String,
            accuracy Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def add_noise(self, inputs, noise_level=0.1):
        """
        افزودن نویز تصادفی به داده‌های ورودی برای تست استحکام مدل.
        """
        noise = torch.randn_like(inputs) * noise_level
        return inputs + noise

    def test_robustness(self, model, data_loader, noise_level=0.1):
        """
        ارزیابی استحکام مدل در برابر تغییرات ورودی‌ها.
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                perturbed_inputs = self.add_noise(inputs, noise_level)
                outputs = model(perturbed_inputs)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def log_robustness_results(self, model_name, version, perturbation_type, accuracy):
        """
        ثبت نتایج تست استحکام مدل در ClickHouse.
        """
        query = f"""
        INSERT INTO model_robustness (model_name, version, perturbation_type, accuracy)
        VALUES ('{model_name}', '{version}', '{perturbation_type}', {accuracy});
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ نتایج تست استحکام مدل {model_name} - نسخه {version} ثبت شد.")

    def get_robustness_history(self, model_name, version, limit=10):
        """
        دریافت تاریخچه‌ی تست استحکام مدل در ClickHouse.
        """
        query = f"""
        SELECT perturbation_type, accuracy, timestamp
        FROM model_robustness
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
    robustness_tester = RobustnessTester()

    # داده‌های ساختگی برای تست
    test_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # ارزیابی استحکام مدل
    accuracy = robustness_tester.test_robustness(model, test_loader, noise_level=0.1)

    # ثبت نتایج تست استحکام در ClickHouse
    robustness_tester.log_robustness_results("TestModel", "1.0", "Gaussian Noise", accuracy)

    # دریافت تاریخچه تست استحکام
    history = robustness_tester.get_robustness_history("TestModel", "1.0")
    print("📊 تاریخچه‌ی تست استحکام مدل:", history)
