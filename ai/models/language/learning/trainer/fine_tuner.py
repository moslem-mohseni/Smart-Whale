import time
import torch
from ai.models.language.infrastructure.redis_connector import RedisConnector
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class FineTuner:
    """
    کلاس FineTuner برای تنظیم دقیق مدل‌های زبانی، مدیریت نرخ یادگیری، و ذخیره اطلاعات Fine-Tuning در ClickHouse.
    """

    def __init__(self, model):
        self.model = model

        # استفاده از سرویس‌های infrastructure
        self.redis_client = RedisConnector()
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ذخیره اطلاعات Fine-Tuning در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره اطلاعات فرآیند Fine-Tuning.
        """
        query = """
        CREATE TABLE IF NOT EXISTS fine_tuning_history (
            model_name String,
            version String,
            learning_rate Float32,
            batch_size Int32,
            epochs Int32,
            optimizer String,
            final_loss Float32,
            final_accuracy Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def get_best_hyperparameters(self, model_name):
        """
        بازیابی بهترین تنظیمات Hyperparameter از Redis یا ClickHouse برای جلوگیری از تکرار تنظیمات ناموفق.
        """
        redis_key = f"fine_tuning:{model_name}"
        cached_params = self.redis_client.get(redis_key)

        if cached_params:
            return eval(cached_params)  # تبدیل مقدار کش‌شده به دیکشنری

        query = f"""
        SELECT learning_rate, batch_size, epochs, optimizer
        FROM fine_tuning_history
        WHERE model_name = '{model_name}'
        ORDER BY final_accuracy DESC
        LIMIT 1;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            best_params = {
                "learning_rate": result[0][0],
                "batch_size": result[0][1],
                "epochs": result[0][2],
                "optimizer": result[0][3]
            }
            self.redis_client.set(redis_key, str(best_params), ex=86400)  # کش یک‌روزه
            return best_params

        return None

    def adjust_learning_rate(self, optimizer, loss_history):
        """
        تنظیم تطبیقی نرخ یادگیری بر اساس روند Loss در طول زمان.
        """
        if len(loss_history) > 1 and loss_history[-1] > loss_history[-2]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9  # کاهش نرخ یادگیری در صورت افزایش Loss
        return optimizer

    def fine_tune(self, train_loader, val_loader, learning_rate=0.001, batch_size=32, epochs=5, optimizer_type="Adam"):
        """
        اجرای فرآیند Fine-Tuning مدل با تنظیمات مشخص‌شده و ذخیره نتایج در ClickHouse.
        """
        optimizer = getattr(torch.optim, optimizer_type)(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        best_accuracy = 0
        loss_history = []

        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0
            correct = 0
            total = 0

            self.model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = correct / total
            loss_history.append(epoch_loss)

            optimizer = self.adjust_learning_rate(optimizer, loss_history)

            print(f"🔹 Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Time: {time.time() - start_time:.2f}s")

            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy

        # ذخیره‌ی اطلاعات Fine-Tuning در ClickHouse
        self.log_fine_tuning_result(learning_rate, batch_size, epochs, optimizer_type, epoch_loss, best_accuracy)

    def log_fine_tuning_result(self, learning_rate, batch_size, epochs, optimizer, final_loss, final_accuracy):
        """
        ذخیره اطلاعات Fine-Tuning در ClickHouse و کش کردن بهترین تنظیمات در Redis.
        """
        query = f"""
        INSERT INTO fine_tuning_history (model_name, version, learning_rate, batch_size, epochs, optimizer, final_loss, final_accuracy)
        VALUES ('{self.model.__class__.__name__}', '1.0', {learning_rate}, {batch_size}, {epochs}, '{optimizer}', {final_loss}, {final_accuracy});
        """
        self.clickhouse_client.execute_query(query)

        # کش کردن بهترین تنظیمات
        redis_key = f"fine_tuning:{self.model.__class__.__name__}"
        best_params = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": optimizer
        }
        self.redis_client.set(redis_key, str(best_params), ex=86400)  # کش یک‌روزه

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
    fine_tuner = FineTuner(model)

    # داده‌های ساختگی برای تست
    train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    val_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    fine_tuner.fine_tune(train_loader, val_loader)
