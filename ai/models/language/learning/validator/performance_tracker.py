import time
import torch
import psutil
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class PerformanceTracker:
    """
    پایش و نظارت بر عملکرد مدل‌های زبانی در طول یادگیری و استنتاج.
    """

    def __init__(self):
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول برای ذخیره اطلاعات عملکرد مدل در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره متریک‌های عملکرد مدل‌های زبانی.
        """
        query = """
        CREATE TABLE IF NOT EXISTS model_performance (
            model_name String,
            version String,
            execution_time Float32,
            gpu_usage Float32,
            cpu_usage Float32,
            memory_usage Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def measure_performance(self, model, inputs):
        """
        اندازه‌گیری زمان اجرا، مصرف GPU/CPU و حافظه مدل.
        """
        start_time = time.time()

        # بررسی مصرف منابع قبل از اجرا
        initial_gpu_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        initial_cpu_usage = psutil.cpu_percent()
        initial_memory_usage = psutil.virtual_memory().percent

        with torch.no_grad():
            _ = model(inputs)

        execution_time = time.time() - start_time

        # بررسی مصرف منابع بعد از اجرا
        final_gpu_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else initial_gpu_memory
        final_cpu_usage = psutil.cpu_percent()
        final_memory_usage = psutil.virtual_memory().percent

        gpu_usage = final_gpu_memory - initial_gpu_memory
        cpu_usage = final_cpu_usage - initial_cpu_usage
        memory_usage = final_memory_usage - initial_memory_usage

        return execution_time, gpu_usage, cpu_usage, memory_usage

    def log_performance_metrics(self, model_name, version, execution_time, gpu_usage, cpu_usage, memory_usage):
        """
        ثبت متریک‌های عملکرد مدل در ClickHouse.
        """
        query = f"""
        INSERT INTO model_performance (model_name, version, execution_time, gpu_usage, cpu_usage, memory_usage)
        VALUES ('{model_name}', '{version}', {execution_time}, {gpu_usage}, {cpu_usage}, {memory_usage});
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ متریک‌های عملکرد مدل {model_name} - نسخه {version} ثبت شد.")

    def get_performance_history(self, model_name, version, limit=10):
        """
        دریافت تاریخچه‌ی عملکرد مدل در ClickHouse.
        """
        query = f"""
        SELECT execution_time, gpu_usage, cpu_usage, memory_usage, timestamp
        FROM model_performance
        WHERE model_name = '{model_name}' AND version = '{version}'
        ORDER BY timestamp DESC
        LIMIT {limit};
        """
        result = self.clickhouse_client.execute_query(query)
        return result

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
    performance_tracker = PerformanceTracker()

    # داده‌های ساختگی برای تست
    inputs = torch.randn(1, 10)

    # اندازه‌گیری و ثبت عملکرد مدل
    exec_time, gpu, cpu, memory = performance_tracker.measure_performance(model, inputs)
    performance_tracker.log_performance_metrics("TestModel", "1.0", exec_time, gpu, cpu, memory)

    # دریافت تاریخچه عملکرد
    history = performance_tracker.get_performance_history("TestModel", "1.0")
    print("📊 تاریخچه‌ی عملکرد مدل:", history)
