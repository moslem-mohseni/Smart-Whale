import torch
import random
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache

class BatchOptimizer:
    """
    کلاس بهینه‌سازی پردازش دسته‌های داده آموزشی برای افزایش کارایی و کاهش مصرف حافظه.
    """

    def __init__(self, model, batch_sizes=[16, 32, 64, 128], trials=5):
        self.model = model
        self.batch_sizes = batch_sizes
        self.trials = trials  # تعداد تست‌های انجام‌شده برای یافتن بهترین دسته

        # اتصال به ClickHouse و Redis برای ذخیره و کش کردن نتایج
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

        # ایجاد جدول برای ذخیره اطلاعات بهینه‌سازی دسته‌ها
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره نتایج بهینه‌سازی دسته‌ها.
        """
        query = """
        CREATE TABLE IF NOT EXISTS batch_optimization (
            model_name String,
            version String,
            batch_size Int32,
            memory_usage Float32,
            training_speed Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def optimize_batch_size(self):
        """
        انتخاب بهترین اندازه دسته بر اساس مصرف حافظه و سرعت پردازش.
        """
        best_batch_size = None
        best_score = float("-inf")

        for batch_size in self.batch_sizes:
            score = self.evaluate_batch(batch_size)

            if score > best_score:
                best_score = score
                best_batch_size = batch_size

        return best_batch_size, best_score

    def evaluate_batch(self, batch_size):
        """
        ارزیابی یک اندازه دسته خاص با محاسبه مصرف حافظه و سرعت پردازش.
        """
        cache_key = f"batch_opt:{batch_size}"
        cached_result = self.redis_cache.get_cache(cache_key)

        if cached_result:
            return cached_result["score"]

        # شبیه‌سازی مصرف حافظه و سرعت پردازش
        memory_usage = random.uniform(0.5, 2.0) * batch_size  # مصرف حافظه بر حسب MB
        training_speed = random.uniform(0.8, 1.5) / batch_size  # سرعت پردازش بر حسب زمان نسبی

        # امتیاز نهایی (وزن‌دهی مصرف حافظه و سرعت پردازش)
        score = (1 / memory_usage) + training_speed

        # ذخیره نتیجه در ClickHouse و کش کردن مقدار آن در Redis
        self.log_result(batch_size, memory_usage, training_speed, score)

        return score

    def log_result(self, batch_size, memory_usage, training_speed, score):
        """
        ذخیره نتایج تنظیمات دسته در ClickHouse و Redis.
        """
        query = f"""
        INSERT INTO batch_optimization (model_name, version, batch_size, memory_usage, training_speed)
        VALUES ('{self.model.__class__.__name__}', '1.0', {batch_size}, {memory_usage}, {training_speed});
        """
        self.clickhouse_client.execute_query(query)

        # ذخیره نتیجه در Redis برای جلوگیری از پردازش تکراری
        self.redis_cache.set_cache(f"batch_opt:{batch_size}", {"score": score})

        print(f"✅ نتیجه‌ی تنظیم دسته ذخیره شد: Batch Size: {batch_size}, Score: {score}")

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
    batch_optimizer = BatchOptimizer(model, batch_sizes=[16, 32, 64, 128], trials=5)

    # اجرای بهینه‌سازی دسته‌ها
    best_batch_size, best_score = batch_optimizer.optimize_batch_size()
    print(f"🎯 بهترین اندازه دسته: {best_batch_size} با امتیاز: {best_score}")
