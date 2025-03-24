# مستندات ماژول ClickHouse

## هدف ماژول

📂 **مسیر ماژول:** `infrastructure/clickhouse/`

ماژول `ClickHouse` برای مدیریت ارتباط با پایگاه داده ClickHouse طراحی شده و یک معماری لایه‌بندی شده با امکانات کامل را ارائه می‌دهد. این ماژول شامل مکانیزم‌های متنوعی برای:
- مدیریت اتصال به پایگاه داده
- اجرای بهینه کوئری‌ها و تحلیل داده‌ها
- کش‌گذاری نتایج کوئری‌های پرتکرار
- بهینه‌سازی عملکرد و فضای ذخیره‌سازی
- نظارت و مانیتورینگ
- پشتیبان‌گیری و مدیریت چرخه عمر داده‌ها
- امنیت و کنترل دسترسی‌ها
- یکپارچه‌سازی با GraphQL، REST API و پردازش استریم

---

## معرفی ماژول

ClickHouse یک پایگاه داده ستونی و توزیع‌شده با عملکرد بالا برای پردازش تحلیلی آنلاین (OLAP) است. این ماژول یک لایه انتزاعی کامل بین برنامه و ClickHouse ایجاد می‌کند و الگوهای طراحی پیشرفته‌ای مانند:

- **الگوی آداپتور**: برای یکپارچه‌سازی با ClickHouse
- **الگوی مدار شکن (Circuit Breaker)**: برای مدیریت اتصالات در زمان خطا
- **الگوی تلاش مجدد (Retry)**: برای تلاش دوباره عملیات ناموفق
- **الگوی Factory Method**: برای ایجاد آسان اجزای مختلف با تنظیمات درست
- **الگوی استخر اتصالات (Connection Pool)**: برای مدیریت بهینه اتصالات
- **الگوی توزیع بار (Load Balancer)**: برای توزیع بهینه درخواست‌ها بین سرورها

این ماژول طراحی شده تا قابلیت اطمینان، مقیاس‌پذیری و عملکرد بالایی را در کار با ClickHouse فراهم کند.

---

## ساختار ماژول

```bash
clickhouse/
│── __init__.py                        # مقداردهی اولیه و تنظیمات متمرکز ماژول
│
├── adapters/                         # مدیریت اتصال‌ها و آداپتورها
│   ├── __init__.py                   # تعریف factory methods برای ایجاد آداپتورها
│   ├── clickhouse_adapter.py         # آداپتور اصلی ClickHouse
│   ├── connection_pool.py            # مدیریت استخر اتصالات
│   ├── circuit_breaker.py            # پیاده‌سازی الگوی Circuit Breaker
│   ├── retry_mechanism.py            # مکانیزم تلاش مجدد
│   ├── load_balancer.py              # توزیع بار بین سرورها
│
├── config/                           # پیکربندی و تنظیمات
│   ├── __init__.py                   # سازماندهی ماژول تنظیمات
│   ├── config.py                     # کلاس مدیریت تنظیمات متمرکز
│
├── domain/                           # مدل‌های دامنه
│   ├── __init__.py                   # معرفی مدل‌های دامنه
│   ├── models.py                     # تعریف مدل‌های داده‌ای ClickHouse
│
├── exceptions/                       # مدیریت خطاها
│   ├── __init__.py                   # معرفی انواع خطاها
│   ├── base.py                       # کلاس پایه خطاهای ClickHouse
│   ├── connection_errors.py          # خطاهای مرتبط با اتصال
│   ├── query_errors.py               # خطاهای مرتبط با کوئری
│   ├── security_errors.py            # خطاهای امنیتی
│   ├── operational_errors.py         # خطاهای عملیاتی
│
├── integration/                      # یکپارچه‌سازی با سایر سیستم‌ها
│   ├── __init__.py                   # معرفی روش‌های یکپارچه‌سازی
│   ├── graphql_layer.py              # لایه GraphQL برای کوئری‌های تحلیلی
│   ├── rest_api.py                   # پیاده‌سازی REST API
│   ├── stream_processor.py           # پردازش داده‌های استریم
│
├── management/                       # مدیریت داده‌ها و پشتیبان‌گیری
│   ├── __init__.py                   # معرفی ابزارهای مدیریتی
│   ├── backup_manager.py             # مدیریت پشتیبان‌گیری و بازیابی داده‌ها
│   ├── data_lifecycle.py             # مدیریت چرخه عمر داده‌ها
│   ├── migration_manager.py          # مدیریت مهاجرت‌های پایگاه داده
│
├── monitoring/                       # نظارت و مانیتورینگ
│   ├── __init__.py                   # معرفی ابزارهای مانیتورینگ
│   ├── performance_monitor.py        # مانیتورینگ عملکرد
│   ├── health_check.py               # بررسی سلامت سیستم
│   ├── prometheus_exporter.py        # صادرکننده متریک‌ها برای Prometheus
│
├── optimization/                     # بهینه‌سازی
│   ├── __init__.py                   # معرفی ابزارهای بهینه‌سازی
│   ├── cache_manager.py              # مدیریت کش داده‌ها
│   ├── data_compressor.py            # فشرده‌سازی داده‌ها
│   ├── query_optimizer.py            # بهینه‌سازی کوئری‌ها
│
├── security/                         # امنیت و کنترل دسترسی
│   ├── __init__.py                   # معرفی ابزارهای امنیتی
│   ├── access_control.py             # کنترل دسترسی و توکن‌های JWT
│   ├── audit_log.py                  # ثبت رخدادهای امنیتی
│   ├── encryption.py                 # رمزنگاری داده‌های حساس
│
└── service/                          # سرویس‌های سطح بالا
    ├── __init__.py                   # معرفی سرویس‌ها و factory method‌ها
    ├── analytics_cache.py            # کش نتایج تحلیلی
    ├── analytics_service.py          # سرویس تحلیل داده‌ها
```

---

## شرح فایل‌ها و عملکرد آن‌ها

### **۱. آداپتورها (Adapters)**

آداپتورها وظیفه مدیریت ارتباط با پایگاه داده ClickHouse را بر عهده دارند. این لایه شامل موارد زیر است:

#### **۱.۱ `clickhouse_adapter.py` - آداپتور اصلی ClickHouse**

آداپتور اصلی که رابط یکپارچه برای ارتباط با ClickHouse فراهم می‌کند.

**کلاس‌ها و متدها:**

- **`ClickHouseAdapter`**
  - `__init__(custom_config)`: مقداردهی اولیه با پیکربندی‌های لازم
  - `async connect()`: برقراری اتصال اولیه و تست آن
  - `async execute(query, params)`: اجرای کوئری با پشتیبانی از مکانیزم‌های خطایابی
  - `async execute_many(queries)`: اجرای چندین کوئری به صورت همزمان
  - `async health_check()`: بررسی سلامت اتصال و گزارش وضعیت
  - `close()`: بستن اتصالات و آزادسازی منابع

این آداپتور از Circuit Breaker، Retry Mechanism و Connection Pool استفاده می‌کند تا اتصال پایدار و مقاوم در برابر خطا فراهم کند.

#### **۱.۲ `connection_pool.py` - مدیریت استخر اتصالات**

مدیریت گروهی از اتصالات برای استفاده بهینه و کاهش سربار ایجاد اتصال جدید.

**کلاس‌ها و متدها:**

- **`ClickHouseConnectionPool`**
  - `get_connection()`: دریافت یک اتصال از استخر
  - `release_connection(connection)`: بازگرداندن اتصال به استخر
  - `close_all()`: بستن تمام اتصالات
  - `get_stats()`: آمار استفاده از استخر اتصالات

#### **۱.۳ `circuit_breaker.py` - الگوی Circuit Breaker**

جلوگیری از ارسال درخواست‌های مکرر به یک سرویس ناموفق برای کاهش فشار و امکان بازیابی.

**کلاس‌ها و متدها:**

- **`CircuitBreaker`**
  - `execute(func, *args, **kwargs)`: اجرای تابع با کنترل Circuit Breaker
  - `execute_with_retry(func, *args, **kwargs)`: اجرای تابع با مکانیزم Retry

#### **۱.۴ `retry_mechanism.py` - مکانیزم تلاش مجدد**

مکانیزم هوشمند برای تلاش مجدد عملیات در صورت شکست اولیه.

**کلاس‌ها و متدها:**

- **`RetryHandler`**
  - `execute_with_retry(func, *args, **kwargs)`: اجرای تابع با تلاش مجدد در صورت شکست

#### **۱.۵ `load_balancer.py` - توزیع بار بین سرورها**

توزیع بار درخواست‌ها بین چندین سرور ClickHouse با پشتیبانی از استراتژی‌های مختلف.

**کلاس‌ها و متدها:**

- **`ClickHouseLoadBalancer`**
  - `get_connection()`: دریافت اتصال از سرور بهینه با استراتژی‌های مختلف
  - `release_connection(server)`: آزادسازی اتصال از سرور مشخص
  - `close_all_connections()`: بستن تمام اتصالات
  - `get_stats()`: آمار استفاده از Load Balancer

---

### **۲. پیکربندی (Config)**

#### **۲.۱ `config.py` - مدیریت متمرکز تنظیمات**

مدیریت تمام تنظیمات مورد نیاز برای ماژول ClickHouse.

**کلاس‌ها و متدها:**

- **`ClickHouseConfig`**
  - `get_connection_params()`: تولید پارامترهای اتصال برای درایور ClickHouse
  - `get_dsn()`: تولید رشته اتصال (DSN)
  - `get_servers()`: لیست سرورهای ClickHouse
  - `get_circuit_breaker_config()`: تنظیمات Circuit Breaker
  - `get_retry_config()`: تنظیمات Retry
  - `get_security_config()`: تنظیمات امنیتی
  - `get_monitoring_config()`: تنظیمات مانیتورینگ
  - `get_data_management_config()`: تنظیمات مدیریت داده

---

### **۳. مدل‌های دامنه (Domain)**

#### **۳.۱ `models.py` - مدل‌های داده‌ای برای تحلیل**

تعریف مدل‌های داده‌ای برای تعامل با ClickHouse.

**کلاس‌ها:**

- **`AnalyticsQuery`**: مدل کوئری تحلیلی با متن کوئری و پارامترها
- **`AnalyticsResult`**: مدل نتیجه کوئری تحلیلی با داده‌ها و خطای احتمالی

---

### **۴. مدیریت خطاها (Exceptions)**

سیستم جامع خطایابی با سلسله‌مراتب خطاهای مرتبط با ClickHouse.

#### **۴.۱ `base.py` - کلاس پایه خطاها**

- **`ClickHouseBaseError`**: کلاس پایه برای تمام خطاهای ClickHouse

#### **۴.۲ دسته‌بندی خطاها**

- **خطاهای اتصال**: `ConnectionError`, `PoolExhaustedError`, `ConnectionTimeoutError`, `AuthenticationError`
- **خطاهای کوئری**: `QueryError`, `QuerySyntaxError`, `QueryExecutionTimeoutError`, `QueryCancellationError`, `DataTypeError`
- **خطاهای امنیتی**: `SecurityError`, `EncryptionError`, `TokenError`, `PermissionDeniedError`
- **خطاهای عملیاتی**: `OperationalError`, `CircuitBreakerError`, `RetryExhaustedError`, `BackupError`, `DataManagementError`

---

### **۵. یکپارچه‌سازی (Integration)**

ماژول‌هایی برای یکپارچه‌سازی ClickHouse با سایر سیستم‌ها.

#### **۵.۱ `graphql_layer.py` - لایه GraphQL**

رابط GraphQL برای اجرای کوئری‌های تحلیلی.

**کلاس‌ها و متدها:**

- **`GraphQLLayer`**
  - `async resolve_query(query_text, variables)`: اجرای کوئری GraphQL
  - `async get_schema()`: دریافت اسکیمای GraphQL

#### **۵.۲ `rest_api.py` - API REST**

ارائه API REST برای اجرای کوئری‌های تحلیلی.

**کلاس‌ها و متدها:**

- **`RestAPI`**
  - متد مسیر `POST /analytics`: اجرای کوئری تحلیلی
  - متد مسیر `GET /health`: بررسی سلامت سرویس
  - متد مسیر `GET /analytics/cache/stats`: آمار کش
  - متد مسیر `POST /analytics/cache/invalidate`: حذف کش

#### **۵.۳ `stream_processor.py` - پردازش استریم**

پردازش داده‌های استریم و ذخیره‌سازی در ClickHouse.

**کلاس‌ها و متدها:**

- **`StreamProcessor`**
  - `async process_stream_data(table_name, data)`: پردازش و درج داده‌های استریم
  - `async _insert_data_batch(table_name, data_list)`: درج دسته‌ای داده‌ها

---

### **۶. مدیریت داده (Management)**

ابزارهای مدیریت پایگاه داده ClickHouse.

#### **۶.۱ `backup_manager.py` - مدیریت پشتیبان‌گیری**

پشتیبان‌گیری و بازیابی داده‌ها.

**کلاس‌ها و متدها:**

- **`BackupManager`**
  - `async create_backup(table_name, partition)`: ایجاد پشتیبان از جدول
  - `async restore_backup(table_name, backup_file)`: بازیابی داده‌ها از پشتیبان
  - `async list_backups(table_name)`: لیست پشتیبان‌های موجود
  - `async delete_backup(backup_file)`: حذف یک فایل پشتیبان

#### **۶.۲ `data_lifecycle.py` - مدیریت چرخه عمر داده‌ها**

نگهداری بهینه داده‌ها در طول زمان.

**کلاس‌ها و متدها:**

- **`DataLifecycleManager`**
  - `async delete_expired_data(table_name, date_column)`: حذف داده‌های منقضی
  - `async optimize_table(table_name, final, deduplicate)`: بهینه‌سازی جدول
  - `async get_table_size_info(table_name)`: اطلاعات حجم جدول
  - `async analyze_database_size()`: تحلیل اندازه پایگاه داده

#### **۶.۳ `migration_manager.py` - مدیریت مهاجرت‌ها**

مدیریت تغییرات ساختاری در پایگاه داده.

**کلاس‌ها و متدها:**

- **`MigrationManager`**
  - `async initialize()`: آماده‌سازی زیرساخت مهاجرت‌ها
  - `async apply_migration(migration_query, migration_id, migration_name)`: اجرای مهاجرت
  - `async rollback_migration(rollback_query, migration_id)`: بازگردانی مهاجرت
  - `async run_migrations(migrations_folder)`: اجرای مهاجرت‌ها از پوشه

---

### **۷. مانیتورینگ (Monitoring)**

نظارت بر عملکرد و سلامت سیستم.

#### **۷.۱ `health_check.py` - بررسی سلامت**

بررسی سلامت اتصال و پایگاه داده.

**کلاس‌ها و متدها:**

- **`HealthCheck`**
  - `check_database_connection()`: بررسی اتصال به پایگاه داده
  - `check_system_health()`: بررسی کلی سلامت سیستم

#### **۷.۲ `performance_monitor.py` - مانیتورینگ عملکرد**

جمع‌آوری متریک‌های عملکردی.

**کلاس‌ها و متدها:**

- **`PerformanceMonitor`**
  - `collect_metrics()`: جمع‌آوری متریک‌ها
  - `start_monitoring(interval)`: شروع حلقه مانیتورینگ مداوم

#### **۷.۳ `prometheus_exporter.py` - صادرات متریک‌ها**

ارائه متریک‌ها برای Prometheus.

**کلاس‌ها و متدها:**

- **`PrometheusExporter`**
  - `update_metrics(query_time, active_connections, disk_usage)`: بروزرسانی متریک‌ها
  - `start_monitoring(interval)`: شروع صادرسازی متریک‌ها

---

### **۸. بهینه‌سازی (Optimization)**

ابزارهای بهینه‌سازی عملکرد و فضا.

#### **۸.۱ `data_compressor.py` - فشرده‌سازی داده‌ها**

فشرده‌سازی داده‌ها برای بهینه‌سازی فضا.

**کلاس‌ها و متدها:**

- **`DataCompressor`**
  - `async optimize_table(table_name, final, partition, deduplicate)`: بهینه‌سازی جدول
  - `async compress_part(table_name, part_name)`: فشرده‌سازی بخشی از جدول
  - `async get_storage_stats(table_name)`: آمار ذخیره‌سازی
  - `async optimize_all_tables(exclude_tables)`: بهینه‌سازی همه جداول

#### **۸.۲ `query_optimizer.py` - بهینه‌سازی کوئری‌ها**

بهبود کارایی کوئری‌ها.

**کلاس‌ها و متدها:**

- **`QueryOptimizer`**
  - `optimize_query(query)`: بهینه‌سازی اولیه کوئری
  - `async optimize_query_with_column_expansion(query)`: بهینه‌سازی پیشرفته با استخراج ستون‌ها
  - `async analyze_query(query)`: تحلیل کوئری و ارائه پیشنهادات بهینه‌سازی
  - `async execute_optimized_query(query, params)`: اجرای کوئری بهینه‌شده

---

### **۹. امنیت (Security)**

مکانیزم‌های حفاظت از داده‌ها و کنترل دسترسی.

#### **۹.۱ `access_control.py` - کنترل دسترسی**

مدیریت توکن‌های JWT و دسترسی‌ها.

**کلاس‌ها و متدها:**

- **`AccessControl`**
  - `generate_token(username, role, custom_claims)`: تولید توکن JWT
  - `verify_token(token)`: بررسی صحت توکن
  - `get_permissions(token)`: استخراج مجوزهای کاربر
  - `refresh_token(token)`: تجدید توکن قبل از انقضا

#### **۹.۲ `audit_log.py` - ثبت رخدادهای امنیتی**

ثبت فعالیت‌ها و رخدادهای امنیتی.

**کلاس‌ها و متدها:**

- **`AuditLogger`**
  - `log_event(username, action, status, details, source_ip, resource)`: ثبت رخداد
  - `log_security_event(event_type, username, success, details)`: ثبت رخداد امنیتی
  - `get_audit_log_path()`: دریافت مسیر فایل لاگ

#### **۹.۳ `encryption.py` - رمزنگاری داده‌ها**

رمزنگاری داده‌های حساس.

**کلاس‌ها و متدها:**

- **`EncryptionManager`**
  - `encrypt(data)`: رمزنگاری داده
  - `decrypt(encrypted_data)`: رمزگشایی داده
  - `rotate_key(new_key)`: تغییر کلید رمزنگاری

---

### **۱۰. سرویس‌ها (Service)**

سرویس‌های سطح بالا برای تعامل با ClickHouse.

#### **۱۰.۱ `analytics_service.py` - سرویس تحلیل داده‌ها**

رابط یکپارچه برای اجرای کوئری‌های تحلیلی.

**کلاس‌ها و متدها:**

- **`AnalyticsService`**
  - `async execute_analytics_query(query)`: اجرای کوئری تحلیلی
  - `async execute_batch_queries(queries)`: اجرای دسته‌ای کوئری‌ها
  - `async invalidate_cache(query)`: حذف کش
  - `async get_cache_stats()`: دریافت آمار کش

#### **۱۰.۲ `analytics_cache.py` - کش نتایج تحلیلی**

مدیریت کش نتایج کوئری‌ها با استفاده از Redis.

**کلاس‌ها و متدها:**

- **`AnalyticsCache`**
  - `async get_cached_result(query, params)`: دریافت نتیجه کش شده
  - `async set_cached_result(query, result, ttl, params)`: ذخیره نتیجه در کش
  - `async invalidate_cache(query, params)`: حذف یک کش خاص یا تمام کش
  - `async get_stats()`: دریافت آمار کش

---

## نحوه استفاده از ماژول ClickHouse

### **۱. راه‌اندازی اولیه کل ماژول**

برای راه‌اندازی سریع و آسان کل ماژول، می‌توانید از تابع `setup_clickhouse` استفاده کنید:

```python
from infrastructure.clickhouse import setup_clickhouse

# راه‌اندازی کامل با تنظیمات پیش‌فرض
adapter, analytics_service, graphql_layer = setup_clickhouse()

# یا با تنظیمات سفارشی
custom_config = {
    "host": "clickhouse-server.example.com",
    "port": 9000,
    "database": "analytics_db",
    "user": "admin",
    "password": "secure_password"
}
adapter, analytics_service, graphql_layer = setup_clickhouse(custom_config)
```

### **۲. استفاده از آداپتور ClickHouse برای اجرای کوئری‌ها**

```python
from infrastructure.clickhouse.adapters import create_adapter

# ایجاد آداپتور با تنظیمات پیش‌فرض
clickhouse_adapter = create_adapter()

# اجرای یک کوئری ساده
query = "SELECT * FROM users WHERE age > :min_age LIMIT 100"
params = {"min_age": 25}
result = await clickhouse_adapter.execute(query, params)

# اجرای چندین کوئری
queries = [
    "SELECT COUNT(*) FROM users",
    "SELECT AVG(age) FROM users",
    "SELECT COUNT(*) FROM orders"
]
results = await clickhouse_adapter.execute_many(queries)

# بررسی سلامت اتصال
health_status = await clickhouse_adapter.health_check()
```

### **۳. اجرای کوئری‌های تحلیلی با AnalyticsService**

```python
from infrastructure.clickhouse.service import create_analytics_service
from infrastructure.clickhouse.domain.models import AnalyticsQuery

# ایجاد سرویس تحلیلی با تنظیمات پیش‌فرض
analytics_service = create_analytics_service()

# تعریف یک کوئری تحلیلی
query = AnalyticsQuery(
    query_text="SELECT date, COUNT(*) as count FROM events WHERE event_type = :type GROUP BY date ORDER BY date",
    params={"type": "click"}
)

# اجرای کوئری با پشتیبانی از کش و بهینه‌سازی
result = await analytics_service.execute_analytics_query(query)

# دسترسی به نتایج
print(f"تعداد رکوردها: {len(result.data)}")
for row in result.data:
    print(f"تاریخ: {row['date']}, تعداد: {row['count']}")

# حذف کش برای بروزرسانی داده‌ها
await analytics_service.invalidate_cache(query)
```

### **۴. استفاده از GraphQL برای کوئری‌های تحلیلی**

```python
from infrastructure.clickhouse.integration import create_graphql_layer

# ایجاد لایه GraphQL
graphql_layer = create_graphql_layer()

# تعریف یک کوئری GraphQL (در این مثال، فرض می‌کنیم به SQL تبدیل می‌شود)
graphql_query = """
{
  analytics {
    userRegistrations(period: "last_month") {
      date
      count
    }
  }
}
"""

# اجرای کوئری GraphQL
result = await graphql_layer.resolve_query(graphql_query)
```

### **۵. استفاده از REST API برای کوئری‌های تحلیلی**

```python
from infrastructure.clickhouse.integration import create_rest_api
import uvicorn

# ایجاد API
rest_api = create_rest_api()

# دریافت برنامه FastAPI
app = rest_api.get_app()

# اجرای سرور
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

سپس می‌توانید با ارسال درخواست‌های REST به مسیرهای تعریف شده کوئری‌ها را اجرا کنید:

```bash
curl -X POST http://localhost:8000/analytics \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT COUNT(*) FROM users", "params": {}}'
```

### **۶. پردازش داده‌های استریم**

```python
from infrastructure.clickhouse.integration import create_stream_processor

# ایجاد پردازشگر استریم
stream_processor = create_stream_processor()

# پردازش داده‌های دریافتی از استریم کافکا
data = [
    {"user_id": 1, "event": "click", "timestamp": "2023-01-01 10:00:00"},
    {"user_id": 2, "event": "view", "timestamp": "2023-01-01 10:05:00"}
]
await stream_processor.process_stream_data("user_events", data)
```

### **۷. مدیریت پشتیبان‌گیری و بازیابی داده‌ها**

```python
from infrastructure.clickhouse.management import create_backup_manager

# ایجاد مدیریت پشتیبان‌گیری
backup_manager = create_backup_manager()

# ایجاد پشتیبان از یک جدول
backup_file = await backup_manager.create_backup("users")

# لیست تمام پشتیبان‌های موجود
backups = await backup_manager.list_backups()

# بازیابی داده‌ها از پشتیبان
success = await backup_manager.restore_backup("users", backup_file)
```

### **۸. مدیریت چرخه عمر داده‌ها**

```python
from infrastructure.clickhouse.management import create_lifecycle_manager

# ایجاد مدیریت چرخه عمر داده‌ها
lifecycle_manager = create_lifecycle_manager()

# حذف داده‌های قدیمی
deleted_count = await lifecycle_manager.delete_expired_data(
    table_name="events",
    date_column="created_at"
)

# بهینه‌سازی جدول
success = await lifecycle_manager.optimize_table("events")

# دریافت اطلاعات اندازه جداول
size_info = await lifecycle_manager.get_table_size_info("events")
```

### **۹. اجرای مهاجرت‌های پایگاه داده**

```python
from infrastructure.clickhouse.management import create_migration_manager

# ایجاد مدیریت مهاجرت‌ها
migration_manager = create_migration_manager()

# آماده‌سازی جدول مهاجرت‌ها
await migration_manager.initialize()

# اجرای یک مهاجرت
migration_query = """
CREATE TABLE IF NOT EXISTS new_analytics_table (
    id UInt64,
    event_date Date,
    user_id UInt32,
    event_type String
) ENGINE = MergeTree() ORDER BY (event_date, id)
"""
success = await migration_manager.apply_migration(
    migration_query=migration_query,
    migration_id="20230101_001",
    migration_name="create_new_analytics_table"
)

# اجرای تمام مهاجرت‌های موجود در پوشه
successful, failed = await migration_manager.run_migrations("/path/to/migrations")
```

### **۱۰. نظارت و مانیتورینگ**

```python
from infrastructure.clickhouse.monitoring import HealthCheck, start_monitoring

# بررسی سلامت سیستم
health_checker = HealthCheck()
is_healthy = health_checker.check_database_connection()
health_status = health_checker.check_system_health()

# شروع مانیتورینگ خودکار
start_monitoring()
```

### **۱۱. امنیت و کنترل دسترسی**

```python
from infrastructure.clickhouse.security import create_access_control, create_encryption_manager, create_audit_logger

# ایجاد سیستم کنترل دسترسی
access_control = create_access_control()

# تولید توکن دسترسی
token = access_control.generate_token(
    username="admin",
    role="superuser",
    custom_claims={"department": "analytics"}
)

# بررسی صحت توکن
decoded_data = access_control.verify_token(token)

# دریافت مجوزهای کاربر
permissions = access_control.get_permissions(token)

# رمزنگاری داده‌های حساس
encryption_manager = create_encryption_manager()
encrypted_data = encryption_manager.encrypt("sensitive_data")
decrypted_data = encryption_manager.decrypt(encrypted_data)

# ثبت رخدادهای امنیتی
audit_logger = create_audit_logger()
audit_logger.log_event(
    username="admin",
    action="delete_user",
    status="success",
    details="Deleted user ID 12345",
    source_ip="192.168.1.1",
    resource="users"
)
```

## سرویس‌های ماژول ClickHouse برای استفاده در سایر ماژول‌ها

ماژول ClickHouse سرویس‌های متنوعی ارائه می‌دهد که می‌توانند در سایر ماژول‌های پروژه استفاده شوند:

### **۱. اجرای کوئری‌های تحلیلی**

سرویس اصلی برای اجرای کوئری‌های تحلیلی در ClickHouse:

```python
from infrastructure.clickhouse.service import create_analytics_service
from infrastructure.clickhouse.domain.models import AnalyticsQuery

async def analyze_user_activity(user_id: int):
    # ایجاد سرویس تحلیلی
    analytics_service = create_analytics_service()
    
    # اجرای کوئری با پارامترهای امن
    query = AnalyticsQuery(
        query_text="SELECT event_type, COUNT(*) as count FROM user_events WHERE user_id = :user_id GROUP BY event_type",
        params={"user_id": user_id}
    )
    
    result = await analytics_service.execute_analytics_query(query)
    return result.data
```

### **۲. ذخیره‌سازی داده‌های رویداد و لاگ‌ها**

برای ذخیره‌سازی حجم بالای داده‌های رویداد و لاگ‌ها:

```python
from infrastructure.clickhouse.integration import create_stream_processor

class EventLogger:
    def __init__(self):
        self.stream_processor = create_stream_processor()
    
    async def log_events(self, events):
        # پردازش و ذخیره انبوه رویدادها
        await self.stream_processor.process_stream_data("system_events", events)
```

### **۳. مدیریت پشتیبان‌گیری خودکار**

سرویس پشتیبان‌گیری خودکار برای سایر ماژول‌ها:

```python
from infrastructure.clickhouse.management import create_backup_manager
import asyncio

class DatabaseBackupService:
    def __init__(self):
        self.backup_manager = create_backup_manager()
    
    async def start_scheduled_backups(self, tables, interval_hours=24):
        while True:
            for table in tables:
                await self.backup_manager.create_backup(table)
                
            # انتظار تا زمان پشتیبان‌گیری بعدی
            await asyncio.sleep(interval_hours * 3600)
```

### **۴. گزارش‌گیری و داشبورد**

ارائه API برای گزارش‌گیری و داشبوردها:

```python
from infrastructure.clickhouse.integration import create_rest_api
from fastapi import FastAPI, APIRouter

class DashboardService:
    def __init__(self, app: FastAPI):
        self.rest_api = create_rest_api()
        self.router = APIRouter()
        self._setup_routes()
        app.include_router(self.router, prefix="/dashboard")
    
    def _setup_routes(self):
        @self.router.get("/summary")
        async def get_summary():
            # استفاده از سرویس تحلیلی برای تولید خلاصه
            # ...
            return {"status": "success", "data": summary_data}
```

### **۵. خدمات سرویس GraphQL**

یکپارچه‌سازی با GraphQL برای ارائه داده‌های تحلیلی:

```python
from infrastructure.clickhouse.integration import create_graphql_layer

class AnalyticsGraphQLService:
    def __init__(self):
        self.graphql_layer = create_graphql_layer()
    
    async def execute_analytics_query(self, query, variables=None):
        result = await self.graphql_layer.resolve_query(query, variables)
        return result
```

### **۶. سرویس‌های رایج بین ماژول‌ها**

در اینجا چند سرویس پایه ارائه می‌شود که می‌توانند در هر ماژولی استفاده شوند:

#### ۶.۱. سرویس `TimescaleClickhouse` - ارتباط بین Timescale و ClickHouse

```python
from infrastructure.clickhouse.adapters import create_adapter
from infrastructure.timescale.adapters import create_timescale_adapter

class TimescaleClickhouseService:
    """سرویس انتقال داده از Timescale به ClickHouse برای تحلیل‌های سریع"""
    
    def __init__(self):
        self.clickhouse_adapter = create_adapter()
        self.timescale_adapter = create_timescale_adapter()
    
    async def migrate_time_series(self, start_date, end_date, table_name):
        """انتقال داده‌های سری زمانی از Timescale به ClickHouse"""
        # دریافت داده از Timescale
        data = await self.timescale_adapter.fetch_data(
            f"SELECT * FROM {table_name} WHERE time BETWEEN $1 AND $2",
            [start_date, end_date]
        )
        
        # تبدیل داده‌ها به فرمت مناسب برای ClickHouse
        prepared_data = self._transform_data(data)
        
        # ذخیره در ClickHouse
        insert_query = f"""
            INSERT INTO {table_name}_analytics (timestamp, value, metadata)
            VALUES (:timestamp, :value, :metadata)
        """
        for batch in self._chunk_data(prepared_data, 1000):
            await self.clickhouse_adapter.execute_many([
                (insert_query, item) for item in batch
            ])
```

#### ۶.۲. سرویس `AnalyticsExporter` - صادرات داده‌های تحلیلی

```python
from infrastructure.clickhouse.service import create_analytics_service
import csv, os

class AnalyticsExporter:
    """صادرات داده‌های تحلیلی به فرمت CSV"""
    
    def __init__(self):
        self.analytics_service = create_analytics_service()
    
    async def export_to_csv(self, query, params, file_path):
        """صادرات نتایج کوئری به فایل CSV"""
        from infrastructure.clickhouse.domain.models import AnalyticsQuery
        
        result = await self.analytics_service.execute_analytics_query(
            AnalyticsQuery(query_text=query, params=params)
        )
        
        if not result.data:
            return False
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', newline='') as csvfile:
            # استخراج نام ستون‌ها از اولین رکورد
            fieldnames = result.data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in result.data:
                writer.writerow(row)
                
        return True
```

#### ۶.۳. سرویس `HealthMonitoringService` - نظارت بر سلامت سیستم

```python
from infrastructure.clickhouse.monitoring import HealthCheck
import asyncio

class HealthMonitoringService:
    """نظارت بر سلامت سیستم‌های مختلف"""
    
    def __init__(self, notification_service=None):
        self.health_check = HealthCheck()
        self.notification_service = notification_service
        self._monitoring = False
    
    async def start_monitoring(self, interval_seconds=60):
        """شروع نظارت دوره‌ای"""
        self._monitoring = True
        
        while self._monitoring:
            health_status = self.health_check.check_system_health()
            
            if health_status['status'] != 'healthy':
                # ارسال اعلان در صورت مشکل
                if self.notification_service:
                    await self.notification_service.send_alert(
                        title="Database Health Alert",
                        message=f"ClickHouse database is unhealthy: {health_status}"
                    )
            
            await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """توقف نظارت"""
        self._monitoring = False
```

---

