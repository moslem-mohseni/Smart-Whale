import torch
import gc
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache

class MemoryEfficient:
    """
    کلاس بهینه‌سازی مصرف حافظه برای کاهش سربار پردازشی و استفاده بهینه از منابع.
    """

    def __init__(self, model, enable_fp16=True, enable_gradient_checkpointing=True):
        self.model = model
        self.enable_fp16 = enable_fp16  # استفاده از محاسبات نیمه‌دقیق (FP16) برای کاهش مصرف حافظه
        self.enable_gradient_checkpointing = enable_gradient_checkpointing  # فعال‌سازی checkpointing برای کاهش مصرف حافظه در یادگیری عمیق

        # اتصال به ClickHouse و Redis برای ذخیره و کش کردن داده‌های مصرف حافظه
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

        # ایجاد جدول برای ذخیره اطلاعات مصرف حافظه و تحلیل روندها
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره اطلاعات مصرف حافظه.
        """
        query = """
        CREATE TABLE IF NOT EXISTS memory_optimization (
            model_name String,
            version String,
            memory_before Float32,
            memory_after Float32,
            optimization_applied String,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def optimize_memory_usage(self):
        """
        کاهش مصرف حافظه با اعمال تکنیک‌های بهینه‌سازی.
        """
        memory_before = self.get_memory_usage()

        # فعال‌سازی FP16 در صورت نیاز
        if self.enable_fp16:
            self.convert_to_fp16()

        # فعال‌سازی Gradient Checkpointing در صورت نیاز
        if self.enable_gradient_checkpointing:
            self.enable_gradient_checkpointing_mode()

        # اجرای Garbage Collector برای آزادسازی حافظه اضافی
        gc.collect()
        torch.cuda.empty_cache()

        memory_after = self.get_memory_usage()

        # ذخیره نتایج در ClickHouse
        self.log_memory_optimization(memory_before, memory_after)

        return memory_after

    def get_memory_usage(self):
        """
        اندازه‌گیری میزان مصرف حافظه فعلی مدل.
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)  # تبدیل به مگابایت
        return 0.0

    def convert_to_fp16(self):
        """
        تبدیل مدل به حالت FP16 برای کاهش مصرف حافظه.
        """
        self.model.half()
        print("✅ مدل به حالت FP16 تبدیل شد.")

    def enable_gradient_checkpointing_mode(self):
        """
        فعال‌سازی Gradient Checkpointing برای کاهش مصرف حافظه.
        """
        if hasattr(self.model, "apply_gradient_checkpointing"):
            self.model.apply_gradient_checkpointing()
            print("✅ Gradient Checkpointing فعال شد.")

    def log_memory_optimization(self, memory_before, memory_after):
        """
        ذخیره اطلاعات بهینه‌سازی مصرف حافظه در ClickHouse.
        """
        optimization_methods = []
        if self.enable_fp16:
            optimization_methods.append("FP16")
        if self.enable_gradient_checkpointing:
            optimization_methods.append("Gradient Checkpointing")

        query = f"""
        INSERT INTO memory_optimization (model_name, version, memory_before, memory_after, optimization_applied)
        VALUES ('{self.model.__class__.__name__}', '1.0', {memory_before}, {memory_after}, '{", ".join(optimization_methods)}');
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ مصرف حافظه قبل از بهینه‌سازی: {memory_before} MB | بعد از بهینه‌سازی: {memory_after} MB")

# تست عملکرد
if __name__ == "__main__":
    import torch.nn as nn

    class SampleModel(nn.Module):
        def __init__(self):
            super(SampleModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = SampleModel()
    memory_optimizer = MemoryEfficient(model, enable_fp16=True, enable_gradient_checkpointing=True)

    # اجرای بهینه‌سازی مصرف حافظه
    optimized_memory = memory_optimizer.optimize_memory_usage()
    print(f"🎯 مصرف حافظه پس از بهینه‌سازی: {optimized_memory} MB")
