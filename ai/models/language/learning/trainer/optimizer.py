import torch
import psutil
from ai.models.language.infrastructure.redis_connector import RedisConnector
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class Optimizer:
    """
    مدیریت بهینه‌سازی فرآیند یادگیری مدل‌ها، تنظیم نرخ یادگیری، مدیریت مصرف منابع پردازشی و ذخیره متریک‌ها.
    """

    def __init__(self, model):
        self.model = model

        # استفاده از سرویس‌های infrastructure
        self.redis_client = RedisConnector()
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ذخیره متریک‌های مصرف منابع در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره اطلاعات مصرف منابع پردازشی در طول آموزش مدل.
        """
        query = """
        CREATE TABLE IF NOT EXISTS training_resource_metrics (
            model_name String,
            version String,
            gpu_usage Float32,
            cpu_usage Float32,
            memory_usage Float32,
            batch_size Int32,
            learning_rate Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def get_system_resources(self):
        """
        دریافت اطلاعات میزان مصرف منابع سیستم (GPU، CPU، حافظه).
        """
        gpu_usage = torch.cuda.memory_allocated(0) / torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else 0
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        return {
            "gpu_usage": gpu_usage,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage
        }

    def get_best_optimization_params(self, model_name):
        """
        دریافت بهترین تنظیمات بهینه‌سازی از Redis یا ClickHouse.
        """
        redis_key = f"optimization_params:{model_name}"
        cached_params = self.redis_client.get(redis_key)

        if cached_params:
            return eval(cached_params)  # تبدیل مقدار کش‌شده به دیکشنری

        query = f"""
        SELECT batch_size, learning_rate
        FROM training_resource_metrics
        WHERE model_name = '{model_name}'
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            best_params = {
                "batch_size": result[0][0],
                "learning_rate": result[0][1]
            }
            self.redis_client.set(redis_key, str(best_params), ex=86400)  # کش یک‌روزه
            return best_params

        return None

    def adjust_batch_size(self, current_batch_size):
        """
        تنظیم اندازه‌ی batch بر اساس میزان مصرف منابع سیستم.
        """
        resources = self.get_system_resources()
        memory_available = 100 - resources["memory_usage"]

        if memory_available < 20:  # اگر میزان حافظه در دسترس کم باشد
            new_batch_size = max(8, int(current_batch_size * 0.8))
        elif memory_available > 50:  # اگر حافظه کافی موجود باشد
            new_batch_size = int(current_batch_size * 1.2)
        else:
            new_batch_size = current_batch_size

        return new_batch_size

    def adjust_learning_rate(self, optimizer, loss_history):
        """
        تنظیم تطبیقی نرخ یادگیری بر اساس روند Loss در طول زمان.
        """
        if len(loss_history) > 1 and loss_history[-1] > loss_history[-2]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9  # کاهش نرخ یادگیری در صورت افزایش Loss
        return optimizer

    def log_resource_metrics(self, version, batch_size, learning_rate):
        """
        ذخیره اطلاعات مصرف منابع در ClickHouse.
        """
        resources = self.get_system_resources()

        query = f"""
        INSERT INTO training_resource_metrics (model_name, version, gpu_usage, cpu_usage, memory_usage, batch_size, learning_rate)
        VALUES ('{self.model.__class__.__name__}', '{version}', {resources["gpu_usage"]}, {resources["cpu_usage"]}, {resources["memory_usage"]}, {batch_size}, {learning_rate});
        """
        self.clickhouse_client.execute_query(query)

    def optimize_training(self, version, initial_batch_size=32, initial_learning_rate=0.001):
        """
        مدیریت فرآیند بهینه‌سازی آموزش مدل.
        """
        best_params = self.get_best_optimization_params(self.model.__class__.__name__)

        if best_params:
            batch_size = self.adjust_batch_size(best_params["batch_size"])
            learning_rate = best_params["learning_rate"]
        else:
            batch_size = initial_batch_size
            learning_rate = initial_learning_rate

        # ذخیره‌ی بهترین تنظیمات در کش Redis
        redis_key = f"optimization_params:{self.model.__class__.__name__}"
        self.redis_client.set(redis_key, str({"batch_size": batch_size, "learning_rate": learning_rate}), ex=86400)

        # ثبت متریک‌های مصرف منابع در ClickHouse
        self.log_resource_metrics(version, batch_size, learning_rate)

        print(f"✅ فرآیند بهینه‌سازی انجام شد: Batch Size = {batch_size}, Learning Rate = {learning_rate}")
        return batch_size, learning_rate

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
    optimizer = Optimizer(model)

    # اجرای فرآیند بهینه‌سازی آموزش مدل
    optimizer.optimize_training(version="1.1")
