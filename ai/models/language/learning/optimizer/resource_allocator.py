import torch
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache

class ResourceAllocator:
    """
    تخصیص منابع پردازشی (CPU، GPU، حافظه) برای بهینه‌سازی فرآیند یادگیری مدل‌ها.
    """

    def __init__(self, max_cpu=8, max_memory=32, max_gpu=1):
        self.max_cpu = max_cpu  # حداکثر تعداد هسته‌های CPU
        self.max_memory = max_memory  # حداکثر حافظه RAM (برحسب گیگابایت)
        self.max_gpu = max_gpu  # حداکثر تعداد GPU

        # اتصال به ClickHouse و Redis برای مدیریت تخصیص منابع
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

        # ایجاد جدول برای ذخیره تخصیص منابع در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره داده‌های تخصیص منابع.
        """
        query = """
        CREATE TABLE IF NOT EXISTS resource_allocations (
            model_name String,
            version String,
            allocated_cpu Int32,
            allocated_memory Float32,
            allocated_gpu Int32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def allocate_resources(self, model_name, cpu=2, memory=4, gpu=0):
        """
        تخصیص منابع پردازشی به یک مدل مشخص.
        """
        if cpu > self.max_cpu or memory > self.max_memory or gpu > self.max_gpu:
            raise ValueError("🚨 مقدار درخواست‌شده برای تخصیص منابع بیش از حد مجاز است!")

        cache_key = f"resource_alloc:{model_name}"
        allocated_resources = {
            "cpu": cpu,
            "memory": memory,
            "gpu": gpu,
        }

        # ذخیره اطلاعات در کش Redis
        self.redis_cache.set_cache(cache_key, allocated_resources)

        # ثبت تخصیص منابع در ClickHouse
        self.log_resource_allocation(model_name, cpu, memory, gpu)

        return allocated_resources

    def log_resource_allocation(self, model_name, cpu, memory, gpu):
        """
        ثبت تخصیص منابع در ClickHouse برای پایش و تحلیل آینده.
        """
        query = f"""
        INSERT INTO resource_allocations (model_name, version, allocated_cpu, allocated_memory, allocated_gpu)
        VALUES ('{model_name}', '1.0', {cpu}, {memory}, {gpu});
        """
        self.clickhouse_client.execute_query(query)

        print(f"✅ تخصیص منابع برای مدل {model_name}: CPU={cpu}, Memory={memory}GB, GPU={gpu}")

    def get_allocated_resources(self, model_name):
        """
        دریافت منابع اختصاص داده‌شده به یک مدل مشخص.
        """
        cache_key = f"resource_alloc:{model_name}"
        cached_result = self.redis_cache.get_cache(cache_key)

        if cached_result:
            return cached_result

        query = f"""
        SELECT allocated_cpu, allocated_memory, allocated_gpu 
        FROM resource_allocations
        WHERE model_name = '{model_name}'
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            return result[0]

        return None

# تست عملکرد
if __name__ == "__main__":
    allocator = ResourceAllocator(max_cpu=8, max_memory=32, max_gpu=1)

    model_name = "SampleModel"

    # تخصیص منابع به مدل
    allocated = allocator.allocate_resources(model_name, cpu=4, memory=8, gpu=1)
    print(f"🎯 منابع تخصیص‌یافته: {allocated}")

    # دریافت منابع اختصاص داده‌شده به مدل
    allocated_info = allocator.get_allocated_resources(model_name)
    print(f"📊 اطلاعات تخصیص منابع: {allocated_info}")
