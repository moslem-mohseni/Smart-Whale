# مستندات ماژول Redis

## هدف ماژول

📂 **مسیر ماژول:** `infrastructure/redis/`

ماژول `Redis` برای مدیریت کش و ذخیره‌سازی داده‌های موقت در سیستم طراحی شده است. این ماژول امکان ارتباط با Redis را فراهم می‌کند و شامل پیکربندی، مدل‌های داده، سرویس مدیریت کش، امنیت، مقیاس‌پذیری و ابزارهای مانیتورینگ است.

---

## **ساختار ماژول**

```
redis/
    │── __init__.py                # مقداردهی اولیه ماژول
    │
    ├── adapters/
    │   ├── redis_adapter.py       # مدیریت اتصال و عملیات Redis
    │   ├── circuit_breaker.py     # مدیریت خطاهای اتصال
    │   ├── connection_pool.py     # مدیریت Connection Pooling
    │   ├── retry_mechanism.py     # اجرای مجدد درخواست‌های ناموفق
    │
    ├── config/
    │   ├── settings.py            # تنظیمات ارتباط و پیکربندی Redis
    │
    ├── domain/
    │   ├── models.py              # مدل‌های داده‌ای برای Redis
    │   ├── compression.py         # فشرده‌سازی داده‌ها
    │   ├── encryption.py          # رمزنگاری داده‌ها
    │
    ├── scripts/
    │   ├── cleanup.py             # پاکسازی دوره‌ای کلیدهای منقضی‌شده
    │   ├── rate_limiter.py        # کنترل نرخ درخواست‌ها
    │
    ├── service/
    │   ├── cache_service.py       # سرویس مدیریت کش
    │   ├── sharded_cache.py       # شاردینگ برای توزیع داده‌ها
    │   ├── fallback_cache.py      # مکانیزم پشتیبان برای Redis
    │
    ├── monitoring/
    │   ├── health_check.py        # بررسی سلامت Redis
    │   ├── metrics.py             # مانیتورینگ Redis با Prometheus
```

---

## **شرح فایل‌ها و عملکرد آن‌ها**

### **1️⃣ `redis_adapter.py` - مدیریت اتصال و عملیات Redis**
📌 **هدف:** مدیریت اتصال به Redis و اجرای عملیات کش‌گذاری.

#### `RedisAdapter`
- `async connect() -> None`: برقراری اتصال به Redis
- `async disconnect() -> None`: قطع اتصال از Redis
- `async get(key: str) -> Optional[Any]`: دریافت مقدار از کش
- `async set(key: str, value: Any, ttl: Optional[int] = None) -> None`: ذخیره مقدار در کش با زمان انقضا
- `async delete(key: str) -> bool`: حذف کلید از کش
- `async hset(key: str, field: str, value: Any) -> None`: ذخیره مقدار در HashMap
- `async hget(key: str, field: str) -> Optional[Any]`: دریافت مقدار از HashMap
- `async expire(key: str, ttl: int) -> bool`: تنظیم زمان انقضا برای کلید
- `async incr(key: str, amount: int = 1) -> int`: افزایش مقدار عددی کلید

---

### **2️⃣ `settings.py` - تنظیمات ارتباط و پیکربندی**
📌 **هدف:** نگهداری اطلاعات اتصال و تنظیمات عملکردی Redis.

#### `RedisConfig`
- `get_connection_params() -> dict`: تولید پارامترهای اتصال برای `aioredis`
- `get_cluster_params() -> dict`: تولید پارامترهای اتصال برای حالت کلاستر
- `get_sentinel_params() -> dict`: تولید پارامترهای اتصال برای Sentinel

---

### **3️⃣ `cache_service.py` - سرویس مدیریت کش**
📌 **هدف:** ارائه یک رابط ساده‌تر برای مدیریت کش.

#### `CacheService`
- `async connect() -> None`: برقراری اتصال به Redis
- `async get(key: str) -> Optional[Any]`: دریافت مقدار از کش
- `async set(key: str, value: Any, ttl: Optional[int] = None) -> None`: ذخیره مقدار در کش
- `async delete(key: str) -> bool`: حذف مقدار از کش
- `async hset(key: str, field: str, value: Any) -> None`: ذخیره مقدار در HashMap
- `async hget(key: str, field: str) -> Optional[Any]`: دریافت مقدار از HashMap
- `async _periodic_cleanup() -> None`: اجرای عملیات دوره‌ای پاکسازی کش
- `async flush() -> None`: اجرای عملیات برای پاک کردن کل کش

---

### **4️⃣ `sharded_cache.py` - توزیع داده‌ها با Sharding**
📌 **هدف:** توزیع داده‌ها در چندین سرور Redis برای بهبود مقیاس‌پذیری.

#### `ShardedCache`
- `async connect() -> None`: اتصال به تمام شاردها
- `async get(key: str) -> Optional[Any]`: دریافت مقدار از شارد مناسب
- `async set(key: str, value: Any, ttl: Optional[int] = None) -> None`: ذخیره مقدار در شارد مناسب
- `async delete(key: str) -> bool`: حذف مقدار از شارد مناسب

---

## **نحوه استفاده از ماژول در پروژه**

### **۱. راه‌اندازی `cache_service.py`**
```python
from infrastructure.redis.config.settings import RedisConfig
from infrastructure.redis.service.cache_service import CacheService

# تنظیمات اتصال
config = RedisConfig()

# راه‌اندازی سرویس
cache_service = CacheService(config)
await cache_service.connect()
await cache_service.set('test_key', 'test_value', ttl=3600)
value = await cache_service.get('test_key')
await cache_service.disconnect()
```

### **۲. استفاده از `sharded_cache.py`**
```python
from infrastructure.redis.service.sharded_cache import ShardedCache

# راه‌اندازی شاردینگ با چند سرور Redis
sharded_cache = ShardedCache([config1, config2])
await sharded_cache.connect()
await sharded_cache.set('user:123', {'name': 'John'}, ttl=7200)
user_data = await sharded_cache.get('user:123')
await sharded_cache.disconnect()
```

