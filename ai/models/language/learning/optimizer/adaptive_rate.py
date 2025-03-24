import numpy as np
import torch
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache

class AdaptiveRate:
    """
    تنظیم تطبیقی نرخ یادگیری بر اساس تغییرات عملکرد مدل و تحلیل روندهای آموزشی.
    """

    def __init__(self, model, base_lr=0.01, min_lr=0.0001, max_lr=0.1, decay_factor=0.9):
        self.model = model
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.decay_factor = decay_factor  # مقدار کاهش نرخ یادگیری در صورت نیاز

        # اتصال به ClickHouse و Redis برای مدیریت داده‌های نرخ یادگیری
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

        # ایجاد جدول برای ذخیره داده‌های نرخ یادگیری
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره تنظیمات نرخ یادگیری.
        """
        query = """
        CREATE TABLE IF NOT EXISTS learning_rate_adjustments (
            model_name String,
            version String,
            new_learning_rate Float32,
            previous_loss Float32,
            improvement_ratio Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def adjust_learning_rate(self, previous_loss, current_loss):
        """
        تنظیم نرخ یادگیری بر اساس تحلیل تغییرات میزان کاهش خطا (Loss).
        """
        improvement_ratio = (previous_loss - current_loss) / previous_loss if previous_loss > 0 else 0

        # در صورتی که کاهش Loss سریع باشد، نرخ یادگیری کاهش می‌یابد تا همگرایی بهتر شود
        if improvement_ratio > 0.1:
            new_learning_rate = max(self.min_lr, self.base_lr * self.decay_factor)
        # در صورتی که کاهش Loss کم باشد، نرخ یادگیری افزایش می‌یابد
        elif improvement_ratio < 0.01:
            new_learning_rate = min(self.max_lr, self.base_lr / self.decay_factor)
        else:
            new_learning_rate = self.base_lr  # حفظ مقدار فعلی در صورت تغییرات جزئی

        # ذخیره اطلاعات تغییر نرخ یادگیری در ClickHouse و Redis
        self.log_adjustment(new_learning_rate, previous_loss, improvement_ratio)

        return new_learning_rate

    def log_adjustment(self, new_learning_rate, previous_loss, improvement_ratio):
        """
        ذخیره اطلاعات نرخ یادگیری و میزان بهبود در ClickHouse.
        """
        query = f"""
        INSERT INTO learning_rate_adjustments (model_name, version, new_learning_rate, previous_loss, improvement_ratio)
        VALUES ('{self.model.__class__.__name__}', '1.0', {new_learning_rate}, {previous_loss}, {improvement_ratio});
        """
        self.clickhouse_client.execute_query(query)

        # کش کردن آخرین نرخ یادگیری در Redis
        self.redis_cache.set_cache(f"learning_rate:{self.model.__class__.__name__}", {"learning_rate": new_learning_rate})

        print(f"✅ نرخ یادگیری جدید {new_learning_rate} برای مدل {self.model.__class__.__name__} ثبت شد.")

# تست عملکرد
if __name__ == "__main__":
    import torch.nn as nn

    class AdaptiveModel(nn.Module):
        def __init__(self):
            super(AdaptiveModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = AdaptiveModel()
    adaptive_rate_manager = AdaptiveRate(model)

    # تست تنظیم نرخ یادگیری
    new_lr = adaptive_rate_manager.adjust_learning_rate(previous_loss=0.5, current_loss=0.4)
    print(f"🎯 نرخ یادگیری تنظیم شده: {new_lr}")
