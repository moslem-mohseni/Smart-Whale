# مستندات ماژول TimescaleDB

## 🌟 هدف ماژول

📂 **مسیر ماژول:** `infrastructure/timescaledb/`

ماژول `TimescaleDB` برای **مدیریت داده‌های سری‌زمانی** طراحی شده است. این ماژول شامل ابزارهایی برای:
- **ذخیره‌سازی و بازیابی داده‌های سری‌زمانی**
- **بهینه‌سازی عملکرد کوئری‌ها**
- **مقیاس‌پذیری و مدیریت حجم بالای داده‌ها**
- **امنیت و کنترل دسترسی کاربران**
- **مانیتورینگ و نظارت بر عملکرد پایگاه داده**
- **مدیریت خودکار داده‌ها و سیاست‌های نگهداری**

این ماژول به‌صورت **ماژولار و مقیاس‌پذیر** طراحی شده است تا در محیط‌های **پرمصرف و پردازشی** بهترین عملکرد را ارائه دهد.

---

## 📂 **ساختار ماژول**
```
timescaledb/
️\200d📚 __init__.py                   # مقداردهی اولیه ماژول

🔠 adapters/
   👉 repository.py              # اینترفیس عمومی Repository
   👉 timeseries_repository.py   # پیاده‌سازی Repository برای داده‌های سری‌زمانی

🔢 config/
   👉 settings.py                # تنظیمات اتصال و متغیرهای محیطی
   👉 connection_pool.py         # مدیریت Connection Pool
   👉 read_write_split.py        # جداسازی عملیات خواندن و نوشتن

📝 domain/
   👉 models.py                  # مدل‌های داده‌ای TimescaleDB
   👉 value_objects.py           # اشیای مقدار (مانند بازه‌های زمانی)

🔒 security/
   👉 access_control.py          # کنترل دسترسی کاربران (RBAC)
   👉 audit_log.py               # ثبت لاگ‌های امنیتی
   👉 encryption.py              # رمزنگاری داده‌های حساس

🌟 optimization/
   👉 cache_manager.py           # مدیریت کش برای بهینه‌سازی عملکرد
   👉 query_optimizer.py         # تحلیل و بهینه‌سازی کوئری‌ها
   👉 data_compressor.py         # فشرده‌سازی داده‌های قدیمی

🛠 storage/
   👉 timescaledb_storage.py     # لایه ذخیره‌سازی و اجرای کوئری‌ها

💪 scaling/
   👉 replication_manager.py     # مدیریت Replication و High Availability
   👉 partition_manager.py       # مدیریت Sharding و Partitioning

📊 monitoring/
   👉 metrics_collector.py       # جمع‌آوری متریک‌های عملکردی
   👉 slow_query_analyzer.py     # بررسی کوئری‌های کند
   👉 health_check.py            # بررسی سلامت پایگاه داده

🛠 service/
   👉 database_service.py        # سرویس مدیریت TimescaleDB
   👉 continuous_aggregation.py  # مدیریت تجمیع داده‌های سری‌زمانی
   👉 data_retention.py          # مدیریت سیاست‌های نگهداری داده

🔧 scripts/
   👉 cleanup_old_data.py        # حذف داده‌های قدیمی
   👉 backup_restore.py          # مدیریت پشتیبان‌گیری و بازیابی
   👉 analyze_performance.py     # تحلیل عملکرد پایگاه داده

📋 migrations/
   👉 v001_initial_schema.py     # ایجاد جداول اولیه
   👉 v002_add_indexes.py        # ایجاد ایندکس‌های بهینه
   👉 v003_partitioning.py       # پیکربندی Partitioning و Sharding
```

---

## ✅ **ویژگی‌های کلیدی ماژول**
✔ **مدیریت داده‌های سری‌زمانی** – بهینه برای پردازش حجم بالای داده‌های مرتبط با زمان  
✔ **مدیریت مقیاس‌پذیری** – پشتیبانی از Replication و Partitioning  
✔ **بهینه‌سازی عملکرد** – کشینگ هوشمند، بهینه‌سازی کوئری‌ها، و فشرده‌سازی داده‌ها  
✔ **امنیت و کنترل دسترسی** – مدیریت نقش‌های کاربری (RBAC) و رمزنگاری داده‌ها  
✔ **مانیتورینگ و نظارت** – مشاهده سلامت پایگاه داده، کوئری‌های کند، و تحلیل عملکرد  

---


# معرفی بخش‌های مختلف ماژول TimescaleDB

## 📂 **1. Adapters - مدیریت ارتباط با داده‌ها**

### `repository.py`
📌 **هدف:** تعریف یک الگوی پایه برای ارتباط با داده‌ها.

**کلاس‌ها و متدها:**
- `Repository`
  - `async add(entity: T) -> T`: افزودن داده جدید.
  - `async get(id: str) -> Optional[T]`: بازیابی داده بر اساس شناسه.
  - `async update(entity: T) -> Optional[T]`: به‌روزرسانی داده.
  - `async delete(id: str) -> bool`: حذف داده بر اساس شناسه.

### `timeseries_repository.py`
📌 **هدف:** مدیریت ذخیره‌سازی و بازیابی داده‌های سری‌زمانی در TimescaleDB.

**کلاس‌ها و متدها:**
- `TimeSeriesRepository`
  - `async get_range(time_range: TimeRange) -> List[T]`: دریافت داده‌ها در یک بازه زمانی.
  - `async get_aggregated(time_range: TimeRange, interval: str, aggregations: Dict[str, str]) -> List[Dict[str, Any]]`: دریافت داده‌های تجمیع شده.
  - `async get_latest(limit: int = 1) -> List[T]`: دریافت آخرین رکوردهای داده.

---

## ⚙ **2. Config - تنظیمات و مدیریت اتصال**

### `settings.py`
📌 **هدف:** نگهداری تنظیمات مربوط به اتصال به پایگاه داده TimescaleDB.

**کلاس‌ها و متدها:**
- `TimescaleDBConfig`
  - `get_connection_string() -> str`: تولید رشته اتصال به پایگاه داده.

### `connection_pool.py`
📌 **هدف:** مدیریت **Connection Pool** برای بهینه‌سازی اتصالات پایگاه داده.

**کلاس‌ها و متدها:**
- `ConnectionPool`
  - `async initialize() -> None`: ایجاد Connection Pool.
  - `async get_connection(read_only: bool = False) -> asyncpg.Connection`: دریافت اتصال از Pool.
  - `async close() -> None`: بستن تمام اتصال‌ها.

### `read_write_split.py`
📌 **هدف:** جداسازی ترافیک خواندن و نوشتن برای توزیع بار.

**کلاس‌ها و متدها:**
- `ReadWriteSplitter`
  - `async execute_query(query: str, params: Optional[List[Any]] = None) -> List[Any]`: اجرای کوئری با تشخیص خودکار Read/Write.

---

## 📊 **3. Domain - مدل‌های داده‌ای**

### `models.py`
📌 **هدف:** تعریف مدل‌های داده‌ای برای TimescaleDB.

**کلاس‌ها:**
- `TimeSeriesData`: مدل داده‌های سری‌زمانی.
- `TableSchema`: مدل طرح جداول در پایگاه داده.

### `value_objects.py`
📌 **هدف:** نمایش مدل‌های مقدار مانند بازه‌های زمانی.

**کلاس‌ها:**
- `TimeRange`: مدل نمایش یک بازه زمانی.

---

## 🔐 **4. Security - امنیت و کنترل دسترسی**

### `access_control.py`
📌 **هدف:** مدیریت نقش‌های کاربری (RBAC) برای دسترسی به پایگاه داده.

**کلاس‌ها و متدها:**
- `AccessControl`
  - `async create_role(role_name: str, privileges: List[str])`: ایجاد یک نقش جدید.
  - `async assign_role_to_user(username: str, role_name: str)`: اختصاص نقش به کاربر.
  - `async check_user_privileges(username: str) -> List[str]`: بررسی سطح دسترسی کاربران.

### `audit_log.py`
📌 **هدف:** ثبت لاگ‌های امنیتی کاربران و عملیات‌های حساس.

**کلاس‌ها و متدها:**
- `AuditLog`
  - `async log_action(username: str, action: str, details: Optional[Dict[str, Any]])`: ثبت لاگ.
  - `async get_logs(username: Optional[str], start_time: Optional[str], end_time: Optional[str]) -> List[Dict[str, Any]]`: دریافت لاگ‌های امنیتی.

### `encryption.py`
📌 **هدف:** رمزنگاری و رمزگشایی داده‌های حساس.

**کلاس‌ها و متدها:**
- `EncryptionManager`
  - `encrypt(plain_text: str) -> str`: رمزنگاری مقدار داده‌شده.
  - `decrypt(encrypted_text: str) -> Optional[str]`: رمزگشایی مقدار رمزنگاری‌شده.

---


# نحوه استفاده از ماژول TimescaleDB

## 🚀 **۱. راه‌اندازی و مقداردهی اولیه ماژول**
برای استفاده از این ماژول، ابتدا باید **تنظیمات اتصال به پایگاه داده** را مقداردهی کنید و سرویس‌های مورد نیاز را مقداردهی اولیه نمایید.

```python
import asyncio
from infrastructure.timescaledb.config.settings import TimescaleDBConfig
from infrastructure.timescaledb.config.connection_pool import ConnectionPool

# مقداردهی اولیه تنظیمات
config = TimescaleDBConfig()

# مقداردهی اولیه Connection Pool
connection_pool = ConnectionPool(config)

async def initialize_db():
    await connection_pool.initialize()
    print("✅ Connection Pool مقداردهی شد.")

asyncio.run(initialize_db())
```

---

## 🗄 **۲. ذخیره‌سازی و واکشی داده‌های سری‌زمانی**
### 📥 **ذخیره‌سازی داده‌های جدید**
```python
from datetime import datetime
from infrastructure.timescaledb.service.database_service import DatabaseService
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

# مقداردهی اولیه سرویس پایگاه داده
storage = TimescaleDBStorage(connection_pool)
db_service = DatabaseService(storage)

async def store_data():
    await db_service.store_time_series_data("time_series_data", 1, datetime.utcnow(), 25.6, {"sensor": "temperature"})
    print("✅ داده با موفقیت ذخیره شد.")

asyncio.run(store_data())
```

### 📤 **دریافت داده‌های سری‌زمانی در یک بازه مشخص**
```python
async def fetch_data():
    from datetime import timedelta
    start_time = datetime.utcnow() - timedelta(days=1)
    end_time = datetime.utcnow()
    data = await db_service.get_time_series_data("time_series_data", start_time, end_time)
    print("📊 داده‌های سری‌زمانی:", data)

asyncio.run(fetch_data())
```

---

## 🛠 **۳. بهینه‌سازی کوئری‌ها و کشینگ**
### ⚡ **استفاده از کش برای بهینه‌سازی کوئری‌ها**
```python
from infrastructure.timescaledb.optimization.cache_manager import CacheManager
from infrastructure.redis.service.cache_service import CacheService
from infrastructure.redis.config.settings import RedisConfig

# مقداردهی اولیه کش
redis_config = RedisConfig()
cache_service = CacheService(redis_config)
await cache_service.connect()
cache_manager = CacheManager(cache_service)

async def cached_query():
    query = "SELECT * FROM time_series_data WHERE timestamp > NOW() - INTERVAL '1 day'"
    result = await cache_manager.get_cached_result(query)
    if not result:
        result = await db_service.execute_query(query)
        await cache_manager.cache_result(query, result)
    print("⚡ داده از کش یا پایگاه داده:", result)

asyncio.run(cached_query())
```

---

## 🔍 **۴. نظارت و مانیتورینگ پایگاه داده**
### 📊 **بررسی وضعیت سلامت پایگاه داده**
```python
from infrastructure.timescaledb.monitoring.health_check import HealthCheck

health_check = HealthCheck(storage)

async def check_health():
    health_status = await health_check.check_health()
    print("🔍 وضعیت سلامت پایگاه داده:", health_status)

asyncio.run(check_health())
```

### 🐢 **بررسی کوئری‌های کند**
```python
from infrastructure.timescaledb.monitoring.slow_query_analyzer import SlowQueryAnalyzer

slow_query_analyzer = SlowQueryAnalyzer(storage)

async def analyze_slow_queries():
    slow_queries = await slow_query_analyzer.get_slow_queries()
    print("🐢 کوئری‌های کند:", slow_queries)

asyncio.run(analyze_slow_queries())
```

---

## 🔐 **۵. مدیریت امنیت و کنترل دسترسی**
### 🛑 **ایجاد نقش کاربری و اختصاص مجوزها**
```python
from infrastructure.timescaledb.security.access_control import AccessControl

access_control = AccessControl(storage)

async def create_role():
    await access_control.create_role("read_only", ["SELECT"])
    await access_control.assign_role_to_user("john_doe", "read_only")
    print("✅ نقش کاربری ایجاد شد.")

asyncio.run(create_role())
```

### 🔑 **رمزنگاری داده‌های حساس**
```python
from infrastructure.timescaledb.security.encryption import EncryptionManager

encryption_manager = EncryptionManager()

secure_data = encryption_manager.encrypt("مقدار حساس")
print("🔐 داده رمزنگاری‌شده:", secure_data)

original_data = encryption_manager.decrypt(secure_data)
print("🔓 داده رمزگشایی‌شده:", original_data)
```

---

## 🔄 **۶. مدیریت داده‌ها و نگهداری بلندمدت**
### 🗑 **حذف داده‌های قدیمی**
```python
from infrastructure.timescaledb.service.data_retention import DataRetention

data_retention = DataRetention(storage)

async def cleanup_data():
    await data_retention.apply_retention_policy("time_series_data")
    print("🗑 داده‌های قدیمی حذف شدند.")

asyncio.run(cleanup_data())
```

### 📦 **پشتیبان‌گیری و بازیابی داده‌ها**
```bash
python infrastructure/timescaledb/scripts/backup_restore.py --backup
python infrastructure/timescaledb/scripts/backup_restore.py --restore /path/to/backup.sql
```

---








