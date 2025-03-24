# مستندات ماژول Core در Smart Whale AI

## 📌 مقدمه
ماژول Core قلب تپنده سیستم AI است که وظیفه ارائه امکانات پایه و مشترک را بر عهده دارد. این ماژول با تمرکز بر اصل "کمترین پردازش و بهترین نتیجه" طراحی شده و مجموعه‌ای از سرویس‌های کلیدی را در اختیار سایر ماژول‌ها قرار می‌دهد.

## 🏗️ ساختار ماژول
```
core/
├── cache/                           # سیستم مدیریت یکپارچه کش
│   ├── hierarchical/
│   │   ├── l1_cache.py             # کش سریع برای داده‌های حیاتی
│   │   ├── l2_cache.py             # کش میانی برای داده‌های عمومی
│   │   └── l3_cache.py             # کش با TTL طولانی
│   ├── manager/
│   │   ├── cache_manager.py        # مدیریت کلی سیستم کش
│   │   ├── invalidation.py         # مکانیزم‌های حذف کش
│   │   └── distribution.py         # توزیع کش بین نودها
│   └── analytics/
│       ├── cache_analyzer.py        # تحلیل کارایی کش
│       └── usage_tracker.py         # ردیابی استفاده از کش
│
├── resource_management/             # مدیریت متمرکز منابع
│   ├── allocator/
│   │   ├── cpu_allocator.py        # تخصیص منابع CPU
│   │   ├── memory_allocator.py     # تخصیص حافظه
│   │   └── gpu_allocator.py        # تخصیص منابع GPU
│   ├── monitor/
│   │   ├── resource_monitor.py     # پایش مصرف منابع
│   │   ├── threshold_manager.py    # مدیریت آستانه‌ها
│   │   └── alert_generator.py      # تولید هشدارها
│   └── optimizer/
│       ├── resource_optimizer.py    # بهینه‌سازی مصرف منابع
│       └── load_balancer.py        # توزیع بار بین منابع
│
├── resilience/                      # مکانیزم‌های تحمل خطا
│   ├── circuit_breaker/
│   │   ├── breaker_manager.py      # مدیریت Circuit Breaker
│   │   ├── state_manager.py        # مدیریت وضعیت‌ها
│   │   └── recovery.py             # مکانیزم‌های بازیابی
│   ├── retry/
│   │   ├── retry_manager.py        # مدیریت تلاش مجدد
│   │   └── backoff_strategies.py   # استراتژی‌های تاخیر
│   └── fallback/
│       ├── fallback_manager.py     # مدیریت جایگزین‌ها
│       └── degradation.py          # کاهش سطح سرویس
│
├── monitoring/                      # سیستم مانیتورینگ
│   ├── metrics/
│   │   ├── collector.py            # جمع‌آوری متریک‌ها
│   │   ├── aggregator.py           # تجمیع متریک‌ها
│   │   └── exporter.py             # صادرکننده متریک‌ها
│   ├── health/
│   │   ├── checker.py              # بررسی سلامت سیستم
│   │   └── reporter.py             # گزارش‌دهی وضعیت
│   └── visualization/
│       ├── dashboard_generator.py   # تولید داشبورد
│       └── alert_visualizer.py      # نمایش هشدارها
│
├── logging/                         # سیستم لاگینگ
│   ├── handlers/
│   │   ├── file_handler.py         # ثبت لاگ در فایل
│   │   ├── kafka_handler.py        # ارسال لاگ به کافکا
│   │   └── elastic_handler.py      # ارسال لاگ به Elasticsearch
│   ├── formatters/
│   │   ├── json_formatter.py       # فرمت JSON
│   │   └── text_formatter.py       # فرمت متنی
│   └── processors/
│       ├── log_processor.py        # پردازش لاگ‌ها
│       └── sensitive_data.py       # حذف داده‌های حساس
│
├── security/                        # امنیت و کنترل دسترسی
│   ├── authentication/
│   │   ├── auth_manager.py         # مدیریت احراز هویت
│   │   └── token_manager.py        # مدیریت توکن‌ها
│   ├── authorization/
│   │   ├── permission_manager.py   # مدیریت مجوزها
│   │   └── role_manager.py         # مدیریت نقش‌ها
│   └── encryption/
│       ├── data_encryptor.py       # رمزنگاری داده‌ها
│       └── key_manager.py          # مدیریت کلیدها
│
└── utils/                          # ابزارهای عمومی
    ├── time/
    │   ├── time_utils.py           # توابع مدیریت زمان
    │   └── scheduler.py            # زمانبندی وظایف
    ├── validation/
    │   ├── input_validator.py      # اعتبارسنجی ورودی
    │   └── schema_validator.py     # اعتبارسنجی ساختار
    └── helpers/
        ├── string_utils.py         # توابع کار با رشته
        └── math_utils.py           # توابع ریاضی
```

## 📦 شرح تفصیلی بخش‌ها

### 1️⃣ سیستم مدیریت یکپارچه کش (`cache/`)

این سیستم با هدف بهینه‌سازی دسترسی به داده‌ها طراحی شده و شامل سه لایه اصلی است:

#### لایه‌های کش (`hierarchical/`)
- **L1 Cache**: کش سریع برای داده‌های حیاتی با TTL کوتاه
  - متدهای اصلی: `fast_get`, `fast_set`, `check_vitality`
  - مکانیزم: استفاده از حافظه محلی برای بالاترین سرعت

- **L2 Cache**: کش میانی برای داده‌های عمومی
  - متدهای اصلی: `distributed_get`, `distributed_set`, `sync_nodes`
  - مکانیزم: توزیع داده بین نودها با Redis

- **L3 Cache**: کش با TTL طولانی برای داده‌های کم‌مصرف
  - متدهای اصلی: `persistent_get`, `persistent_set`, `cleanup`
  - مکانیزم: ذخیره‌سازی با قابلیت persistence

#### مدیریت کش (`manager/`)
- **Cache Manager**: مدیریت یکپارچه لایه‌های کش
  - متدهای اصلی: `smart_get`, `smart_set`, `promote`, `demote`
  - مکانیزم: تصمیم‌گیری هوشمند برای انتخاب لایه مناسب

- **Invalidation**: مدیریت اعتبار کش
  - متدهای اصلی: `invalidate`, `batch_invalidate`, `pattern_invalidate`
  - مکانیزم: حذف هوشمند داده‌های منقضی

- **Distribution**: توزیع کش بین نودها
  - متدهای اصلی: `distribute`, `rebalance`, `sync`
  - مکانیزم: توزیع متوازن داده‌ها بین نودهای مختلف

#### تحلیل کش (`analytics/`)
- **Cache Analyzer**: تحلیل کارایی سیستم کش
  - متدهای اصلی: `analyze_hit_rate`, `analyze_memory_usage`, `optimize`
  - مکانیزم: تحلیل مداوم و ارائه پیشنهادات بهینه‌سازی

- **Usage Tracker**: ردیابی الگوهای استفاده
  - متدهای اصلی: `track_usage`, `predict_usage`, `generate_insights`
  - مکانیزم: جمع‌آوری و تحلیل الگوهای استفاده

### 2️⃣ مدیریت متمرکز منابع (`resource_management/`)

این بخش مسئول مدیریت بهینه منابع سیستم است:

#### تخصیص‌دهنده منابع (`allocator/`)
- **CPU Allocator**: مدیریت منابع CPU
  - متدهای اصلی: `allocate_cpu`, `optimize_threads`, `manage_priority`
  - مکانیزم: تخصیص هوشمند هسته‌های CPU

- **Memory Allocator**: مدیریت حافظه
  - متدهای اصلی: `allocate_memory`, `defragment`, `optimize_usage`
  - مکانیزم: مدیریت پویای حافظه و جلوگیری از نشتی

- **GPU Allocator**: مدیریت منابع GPU
  - متدهای اصلی: `allocate_gpu`, `share_resources`, `optimize_workload`
  - مکانیزم: تخصیص بهینه منابع GPU

#### پایش منابع (`monitor/`)
- **Resource Monitor**: پایش مصرف منابع
  - متدهای اصلی: `monitor_usage`, `detect_anomalies`, `predict_needs`
  - مکانیزم: پایش مداوم و پیش‌بینی نیازها

- **Threshold Manager**: مدیریت آستانه‌ها
  - متدهای اصلی: `set_thresholds`, `adjust_dynamic`, `handle_breach`
  - مکانیزم: تنظیم پویای آستانه‌های هشدار

#### بهینه‌سازی (`optimizer/`)
- **Resource Optimizer**: بهینه‌سازی مصرف منابع
  - متدهای اصلی: `optimize_allocation`, `balance_load`, `reduce_waste`
  - مکانیزم: بهینه‌سازی مداوم تخصیص منابع

### 3️⃣ مکانیزم‌های تحمل خطا (`resilience/`)

این بخش مسئول حفظ پایداری سیستم است:

#### Circuit Breaker (`circuit_breaker/`)
- **Breaker Manager**: مدیریت Circuit Breaker
  - متدهای اصلی: `check_state`, `trip_breaker`, `reset_breaker`
  - مکانیزم: جلوگیری از فراخوانی‌های ناموفق

- **State Manager**: مدیریت وضعیت‌ها
  - متدهای اصلی: `manage_state`, `transition`, `store_history`
  - مکانیزم: نگهداری و مدیریت وضعیت‌های مختلف

#### Retry Management (`retry/`)
- **Retry Manager**: مدیریت تلاش مجدد
  - متدهای اصلی: `attempt_retry`, `calculate_delay`, `track_attempts`
  - مکانیزم: تلاش مجدد هوشمند با تاخیر مناسب

### 4️⃣ مانیتورینگ (`monitoring/`)

سیستم جامع پایش و گزارش‌گیری:

#### متریک‌ها (`metrics/`)
- **Collector**: جمع‌آوری متریک‌ها
  - متدهای اصلی: `collect_metrics`, `process_data`, `store_metrics`
  - مکانیزم: جمع‌آوری مداوم متریک‌های مختلف

- **Aggregator**: تجمیع متریک‌ها
  - متدهای اصلی: `aggregate`, `calculate_statistics`, `generate_summaries`
  - مکانیزم: تجمیع و پردازش آماری داده‌ها

### 5️⃣ لاگینگ (`logging/`)

سیستم ثبت و مدیریت لاگ‌ها:

#### Handlers (`handlers/`)
- **File Handler**: ثبت لاگ در فایل
  - متدهای اصلی: `write_log`, `rotate_files`, `cleanup_old`
  - مکانیزم: مدیریت فایل‌های لاگ با چرخش خودکار

#### Processors (`processors/`)
- **Log Processor**: پردازش لاگ‌ها
  - متدهای اصلی: `process_log`, `filter_sensitive`, `enrich_data`
  - مکانیزم: پردازش و غنی‌سازی لاگ‌ها

### 6️⃣ امنیت (`security/`)

مدیریت امنیت و کنترل دسترسی:

#### Authentication (`authentication/`)
- **Auth Manager**: مدیریت احراز هویت
  - متدهای اصلی:
    - `authenticate`: احراز هویت کاربر با روش‌های مختلف
    - `verify_token`: بررسی اعتبار توکن‌های دسترسی
    - `manage_sessions`: مدیریت نشست‌های کاربری
  - مکانیزم: پیاده‌سازی استراتژی‌های مختلف احراز هویت با قابلیت توسعه

- **Token Manager**: مدیریت توکن‌ها
  - متدهای اصلی:
    - `generate_token`: تولید توکن با الگوریتم‌های امن
    - `validate_token`: اعتبارسنجی توکن‌ها
    - `refresh_token`: تمدید خودکار توکن‌ها
  - مکانیزم: مدیریت چرخه حیات توکن‌ها با امنیت بالا

#### Authorization (`authorization/`)
- **Permission Manager**: مدیریت مجوزها
  - متدهای اصلی:
    - `check_permission`: بررسی دسترسی به منابع
    - `grant_permission`: اعطای مجوز
    - `revoke_permission`: لغو مجوز
  - مکانیزم: کنترل دسترسی مبتنی بر نقش (RBAC) با قابلیت تعریف مجوزهای سفارشی

- **Role Manager**: مدیریت نقش‌ها
  - متدهای اصلی:
    - `manage_roles`: مدیریت نقش‌های سیستم
    - `assign_role`: تخصیص نقش به کاربر
    - `check_role_hierarchy`: بررسی سلسله مراتب نقش‌ها
  - مکانیزم: پیاده‌سازی سیستم نقش‌های چندسطحی با وراثت

#### Encryption (`encryption/`)
- **Data Encryptor**: رمزنگاری داده‌ها
  - متدهای اصلی:
    - `encrypt_data`: رمزنگاری داده‌ها با الگوریتم‌های استاندارد
    - `decrypt_data`: رمزگشایی داده‌ها
    - `verify_integrity`: بررسی یکپارچگی داده‌ها
  - مکانیزم: استفاده از الگوریتم‌های رمزنگاری قوی با مدیریت کلید امن

- **Key Manager**: مدیریت کلیدها
  - متدهای اصلی:
    - `generate_keys`: تولید کلیدهای رمزنگاری
    - `rotate_keys`: چرخش دوره‌ای کلیدها
    - `store_keys`: ذخیره‌سازی امن کلیدها
  - مکانیزم: مدیریت چرخه حیات کلیدهای رمزنگاری با امنیت بالا

### 7️⃣ ابزارهای عمومی (`utils/`)

این بخش شامل مجموعه‌ای از ابزارها و توابع کمکی است که توسط سایر بخش‌ها مورد استفاده قرار می‌گیرند:

#### مدیریت زمان (`time/`)
- **Time Utils**: توابع کار با زمان
  - متدهای اصلی:
    - `convert_timezone`: تبدیل منطقه‌های زمانی
    - `format_datetime`: قالب‌بندی تاریخ و زمان
    - `calculate_duration`: محاسبه مدت زمان
  - مکانیزم: مدیریت یکپارچه زمان در کل سیستم

- **Scheduler**: زمانبندی وظایف
  - متدهای اصلی:
    - `schedule_task`: زمانبندی اجرای وظایف
    - `manage_recurring`: مدیریت وظایف تکرارشونده
    - `handle_conflicts`: مدیریت تداخل‌های زمانی
  - مکانیزم: زمانبندی هوشمند با در نظر گرفتن بار سیستم

#### اعتبارسنجی (`validation/`)
- **Input Validator**: اعتبارسنجی ورودی‌ها
  - متدهای اصلی:
    - `validate_input`: اعتبارسنجی داده‌های ورودی
    - `sanitize_data`: پاکسازی داده‌ها
    - `check_constraints`: بررسی محدودیت‌ها
  - مکانیزم: اعتبارسنجی جامع با قوانین قابل تعریف

- **Schema Validator**: اعتبارسنجی ساختار
  - متدهای اصلی:
    - `validate_schema`: بررسی ساختار داده‌ها
    - `check_compatibility`: بررسی سازگاری
    - `generate_schema`: تولید خودکار ساختار
  - مکانیزم: اعتبارسنجی ساختاری با پشتیبانی از فرمت‌های مختلف

#### توابع کمکی (`helpers/`)
- **String Utils**: توابع کار با رشته
  - متدهای اصلی:
    - `normalize_text`: نرمال‌سازی متن
    - `find_patterns`: جستجوی الگوها
    - `text_metrics`: محاسبه معیارهای متنی
  - مکانیزم: پردازش متن با پشتیبانی از زبان‌های مختلف

- **Math Utils**: توابع ریاضی
  - متدهای اصلی:
    - `statistical_analysis`: تحلیل‌های آماری
    - `numerical_operations`: عملیات عددی
    - `matrix_calculations`: محاسبات ماتریسی
  - مکانیزم: محاسبات ریاضی بهینه‌شده

## 🔄 تعامل بین اجزا

### 1. تعامل سیستم کش با مدیریت منابع
- بهینه‌سازی خودکار حافظه اختصاص یافته به کش
- تنظیم پویای سایز کش بر اساس بار سیستم
- مدیریت هوشمند invalidation برای کاهش مصرف منابع

### 2. تعامل مانیتورینگ با سایر بخش‌ها
- جمع‌آوری متریک‌های عملکردی از تمام اجزا
- ارائه دید یکپارچه از وضعیت سیستم
- هشداردهی خودکار در صورت مشاهده ناهنجاری

### 3. تعامل سیستم امنیت با لاگینگ
- ثبت تمام رویدادهای امنیتی
- پایش الگوهای مشکوک
- حفظ حریم خصوصی در لاگ‌ها

## 📊 استراتژی‌های بهینه‌سازی

### 1. بهینه‌سازی حافظه
- استفاده از مکانیزم‌های lazy loading
- مدیریت هوشمند garbage collection
- کش‌گذاری چندسطحی با پیش‌بینی نیازها

### 2. بهینه‌سازی CPU
- استفاده از thread pooling
- پردازش ناهمزمان عملیات غیرضروری
- مدیریت هوشمند priority scheduling

### 3. بهینه‌سازی I/O
- استفاده از non-blocking I/O
- batch processing برای عملیات دیسک
- بافرینگ هوشمند

## 🛡️ استراتژی‌های امنیتی

### 1. امنیت داده
- رمزنگاری end-to-end
- مدیریت امن کلیدها
- حذف امن داده‌های حساس

### 2. کنترل دسترسی
- احراز هویت چندعاملی
- مدیریت جلسات با امنیت بالا
- کنترل دسترسی مبتنی بر نقش

### 3. امنیت عملیاتی
- پایش مداوم فعالیت‌ها
- تشخیص و پاسخ به حملات
- بروزرسانی خودکار تنظیمات امنیتی


# مستندات ماژول Messaging در Smart Whale AI

## 📌 مقدمه
ماژول Messaging، ستون فقرات سیستم ارتباطی Smart Whale AI محسوب می‌شود. این ماژول با استفاده از Kafka، امکان ارسال و دریافت پیام‌ها را بین اجزای مختلف سیستم فراهم می‌کند. با بهره‌گیری از ساختارهای استاندارد پیام، مدیریت موضوعات (Topics) و توابع کمکی جهت اعتبارسنجی، این ماژول تضمین می‌کند که ارتباطات به صورت یکپارچه، امن و با کارایی بالا انجام شوند.

---

## 🏗️ ساختار ماژول
ساختار دایرکتوری ماژول Messaging به صورت زیر است:

دایرکتوری:

ai/core/messaging/
```
messaging/
├── __init__.py                # صادرسازی API‌های اصلی ماژول Messaging
├── constants.py               # ثابت‌ها و تنظیمات پیش‌فرض پیام‌رسانی
├── kafka_service.py           # سرویس ارتباط با Kafka و مدیریت پیام‌ها
├── message_schemas.py         # ساختارهای پیام و قراردادهای ارتباطی
└── topic_manager.py           # مدیریت موضوعات Kafka و ساختاردهی به آن‌ها
```

---

## 📦 شرح تفصیلی بخش‌ها

### 1️⃣ ثابت‌ها و تنظیمات پیام‌رسانی (`constants.py`)
این بخش شامل تمام ثابت‌ها و تنظیمات پیش‌فرض مورد استفاده در ماژول Messaging است. تمامی مقادیر مهم در این فایل تعریف شده‌اند تا یکپارچگی در استفاده از موضوعات و پیام‌ها تضمین شود.

- **نسخه ماژول پیام‌رسانی**  
  - `MESSAGING_VERSION = "1.0.0"`

- **پیشوندهای موضوعات کافکا**  
  - `TOPIC_PREFIX = "smartwhale"`  
  - `DATA_PREFIX = f"{TOPIC_PREFIX}.data"`  
  - `MODELS_PREFIX = f"{TOPIC_PREFIX}.models"`  
  - `BALANCE_PREFIX = f"{TOPIC_PREFIX}.balance"`  
  - `CORE_PREFIX = f"{TOPIC_PREFIX}.core"`

- **نام‌های استاندارد موضوعات**  
  - `DATA_REQUESTS_TOPIC = f"{DATA_PREFIX}.requests"`  
  - `DATA_RESPONSES_PREFIX = f"{DATA_PREFIX}.responses"`  
  - `MODELS_REQUESTS_TOPIC = f"{MODELS_PREFIX}.requests"`  
  - `BALANCE_METRICS_TOPIC = f"{BALANCE_PREFIX}.metrics"`  
  - `BALANCE_EVENTS_TOPIC = f"{BALANCE_PREFIX}.events"`  
  - `SYSTEM_LOGS_TOPIC = f"{CORE_PREFIX}.logs"`

- **تنظیمات پیش‌فرض برای موضوعات**  
  - `DEFAULT_PARTITIONS = 5`  
  - `DEFAULT_REPLICATION = 3`  
  - `MODEL_TOPIC_PARTITIONS = 3`  
  - `MODEL_TOPIC_REPLICATION = 2`

- **مقادیر پیش‌فرض برای اولویت‌ها**  
  - `PRIORITY_CRITICAL = 1`  
  - `PRIORITY_HIGH = 2`  
  - `PRIORITY_MEDIUM = 3`  
  - `PRIORITY_LOW = 4`  
  - `PRIORITY_BACKGROUND = 5`  
  - `DEFAULT_PRIORITY = PRIORITY_MEDIUM`

- **تنظیمات زمان انتظار و تایم‌اوت**  
  - `DEFAULT_OPERATION_TIMEOUT = 30`  
  - `DEFAULT_KAFKA_TIMEOUT = 10`

- **اندازه‌های پیش‌فرض**  
  - `DEFAULT_BATCH_SIZE = 100`  
  - `DEFAULT_BUFFER_SIZE = 1024 * 1024  # 1MB`

- **منابع درخواست**  
  - `REQUEST_SOURCE_USER = "user"`  
  - `REQUEST_SOURCE_MODEL = "model"`  
  - `REQUEST_SOURCE_SYSTEM = "system"`  
  - `DEFAULT_REQUEST_SOURCE = REQUEST_SOURCE_USER`

- **وضعیت‌های پیام**  
  - `MESSAGE_STATUS_SUCCESS = "success"`  
  - `MESSAGE_STATUS_ERROR = "error"`  
  - `MESSAGE_STATUS_PENDING = "pending"`  
  - `MESSAGE_STATUS_PARTIAL = "partial"`  
  - `MESSAGE_STATUS_TIMEOUT = "timeout"`

- **برچسب‌های خطا**  
  - `ERROR_INVALID_REQUEST = "invalid_request"`  
  - `ERROR_NO_DATA_FOUND = "no_data_found"`  
  - `ERROR_SOURCE_UNAVAILABLE = "source_unavailable"`  
  - `ERROR_PROCESSING_FAILED = "processing_failed"`  
  - `ERROR_TIMEOUT = "timeout"`  
  - `ERROR_UNKNOWN = "unknown_error"`

---

### 2️⃣ سرویس کافکا (`kafka_service.py`)
این بخش مسئول برقراری ارتباط با سرور Kafka و مدیریت پیام‌ها می‌باشد. کلاس اصلی موجود در این فایل، **KafkaService**، تمامی عملیات مربوط به ارسال، دریافت و مدیریت پیام‌ها را پیاده‌سازی می‌کند.

#### کلاس KafkaService
- **هدف**:  
  مدیریت ارتباط با Kafka، ارسال پیام‌های تکی و دسته‌ای، اعتبارسنجی و ارسال پیام‌های ساختار یافته (DataRequest و DataResponse) و مدیریت اشتراک‌ها.
  
- **متدهای اصلی**:

  - **`__init__`**  
    مقداردهی اولیه سرویس؛ تنظیم پیکربندی Kafka، تولیدکننده پیام (producer)، نگهداری لیست مصرف‌کننده‌ها (consumers) و وضعیت اتصال.
  
  - **`connect()`**  
    برقراری اتصال با سرور Kafka. در صورت عدم اتصال، یک producer ایجاد می‌شود.  
    _بازگشت: True در صورت موفقیت._

  - **`disconnect()`**  
    قطع اتصال از Kafka و آزادسازی منابع؛ شامل لغو اشتراک‌های فعال و بستن producer.  
    _بازگشت: True در صورت موفقیت._

  - **`send_message(topic, message_data)`**  
    ارسال یک پیام به موضوع مشخص؛ پیام‌ها می‌توانند به صورت دیکشنری، رشته یا بایت باشند.  
    - تبدیل داده به قالب JSON (در صورت نیاز)  
    - ساخت یک شیء Message و ارسال آن  
    _بازگشت: True در صورت موفقیت._

  - **`send_batch(topic, messages)`**  
    ارسال دسته‌ای پیام‌ها به یک موضوع؛ پردازش لیست پیام‌ها و تبدیل هر کدام به قالب مناسب.  
    _بازگشت: True در صورت موفقیت همه ارسال‌ها._

  - **`send_data_request(request, topic)`**  
    ارسال پیام‌های درخواست داده؛ اعتبارسنجی ساختار DataRequest و ارسال آن به موضوع مشخص.  
    _بازگشت: True در صورت موفقیت._

  - **`send_data_response(response, topic)`**  
    ارسال پیام‌های پاسخ داده؛ اعتبارسنجی ساختار DataResponse و ارسال آن به موضوع مشخص.  
    _بازگشت: True در صورت موفقیت._

  - **`subscribe(topic, group_id, handler)`**  
    اشتراک در یک موضوع به منظور دریافت پیام‌ها.  
    - ایجاد مصرف‌کننده جدید  
    - تبدیل پیام‌های دریافتی از قالب Kafka به دیکشنری  
    - فراخوانی تابع handler جهت پردازش پیام‌ها  
    _بازگشت: True در صورت موفقیت._

  - **`unsubscribe(topic, group_id)`**  
    لغو اشتراک از یک موضوع؛ متوقف کردن مصرف‌کننده و پاکسازی اشتراک‌های ذخیره شده.  
    _بازگشت: True در صورت موفقیت._

  - **`topic_exists(topic_name)`**  
    بررسی وجود یک موضوع در Kafka با استفاده از لیست موضوعات.  
    _بازگشت: True در صورت وجود موضوع._

  - **`create_topic(topic_name, num_partitions, replication_factor)`**  
    ایجاد یک موضوع جدید در Kafka در صورتی که موضوع مورد نظر وجود نداشته باشد.  
    _بازگشت: True در صورت موفقیت ایجاد موضوع._

  - **`delete_topic(topic_name)`**  
    حذف یک موضوع از Kafka؛ شامل حذف موضوع از لیست‌های داخلی.  
    _بازگشت: True در صورت موفقیت._

  - **`list_topics()`**  
    دریافت لیست تمام موضوعات موجود در Kafka؛ این متد در پیاده‌سازی فعلی به صورت فرضی است.  
    _بازگشت: لیست نام موضوعات._

  - **`get_topic_info(topic_name)`**  
    دریافت اطلاعات یک موضوع (مانند تعداد پارتیشن‌ها و فاکتور تکرار)؛ در صورت عدم وجود موضوع، مقدار None بازگردانده می‌شود.

- **نمونه Singleton**:  
  - `kafka_service = KafkaService()`

---

### 3️⃣ ساختارهای پیام و قراردادهای ارتباطی (`message_schemas.py`)
در این بخش ساختارهای پیام استاندارد جهت تبادل اطلاعات بین ماژول‌ها تعریف شده‌اند. این ساختارها تضمین می‌کنند که پیام‌ها به صورت یکپارچه و با فرمت مشخص ارسال و دریافت شوند.

#### ۳.۱ انواع داده و منابع
- **DataType**  
  تعریف انواع داده‌های قابل جمع‌آوری مانند: TEXT، IMAGE، VIDEO، AUDIO، DOCUMENT، STRUCTURED و MIXED.

- **DataSource**  
  تعریف منابع داده مانند: WEB، WIKI، TWITTER، TELEGRAM، YOUTUBE، APARAT، NEWS، BOOK، API، LINKEDIN، PODCAST، RSS، SCIENTIFIC، DATABASE و CUSTOM.

- **OperationType**  
  انواع عملیات قابل درخواست مانند: FETCH_DATA، PROCESS_DATA، UPDATE_DATA، DELETE_DATA، MONITOR_DATA، VERIFY_DATA، SEARCH_DATA، TRANSLATE_DATA، SUMMARIZE_DATA و ANALYZE_DATA.

- **Priority**  
  سطوح اولویت پیام‌ها شامل CRITICAL، HIGH، MEDIUM، LOW و BACKGROUND.

- **RequestSource**  
  منابع درخواست شامل USER، MODEL، SYSTEM، SCHEDULED و API.

#### ۳.۲ کلاس‌های ساختار پیام

- **MessageMetadata**  
  ساختار متادیتای پایه برای همه پیام‌ها.  
  - **فیلدها**:
    - `request_id`: شناسه یکتا (با استفاده از uuid).
    - `timestamp`: زمان ارسال پیام (به فرمت ISO).
    - `source`: مبدأ پیام (مثلاً "balance").
    - `destination`: مقصد پیام (مثلاً "data").
    - `priority`: اولویت پیام.
    - `request_source`: منبع درخواست.
    - فیلدهای تکمیلی مانند `correlation_id`, `trace_id`, `session_id`.
    - مدیریت خطا و بازپخش با `retry_count`, `ttl`, `expires_at`.
  - **متدها**:
    - `to_dict()`: تبدیل به دیکشنری.
    - `from_dict()`: ایجاد نمونه از دیکشنری.

- **RequestPayload**  
  ساختار محتوای درخواست ارسال شده از Balance به Data.  
  - **فیلدها**:
    - `operation`: نوع عملیات (مثلاً FETCH_DATA).
    - `model_id`: شناسه مدل درخواست‌کننده.
    - `data_type`: نوع داده مورد نیاز.
    - `data_source`: منبع داده (اختیاری).
    - `parameters`: پارامترهای اختصاصی.
    - `response_topic`: موضوعی که پاسخ باید در آن ارسال شود.
    - تنظیمات اضافی مانند `batch_id`, `timeout`, `max_size`, `require_fresh`.
    - فراداده‌ها مانند `tags` و `context`.
  - **متدها**:
    - `to_dict()`: تبدیل به دیکشنری (با تبدیل Enum به رشته).
    - `from_dict()`: ایجاد نمونه از دیکشنری با پردازش مقادیر Enum.

- **ResponsePayload**  
  ساختار محتوای پاسخ ارسال شده از Data به Models.  
  - **فیلدها**:
    - `operation`: نوع عملیات.
    - `status`: وضعیت پاسخ (مثلاً success یا error).
    - `data_type`: نوع داده.
    - `data_source`: منبع داده (اختیاری).
    - `data`: داده‌های اصلی پاسخ.
    - `error_message` و `error_code` در صورت بروز خطا.
    - اطلاعات تکمیلی مانند `metrics`, `batch_id`, `next_cursor`, `total_items`, `page_info`.
    - فراداده‌ها مانند `metadata`, `source_url`, `last_updated`.
    - اطلاعات پردازشی مانند `processing_time`, `cached`, `additional_info`.
  - **متدها**:
    - `to_dict()`: تبدیل به دیکشنری.
    - `from_dict()`: ساخت نمونه از دیکشنری.

- **DataRequest**  
  کلاس اصلی پیام درخواست؛ ترکیبی از `MessageMetadata` و `RequestPayload`.  
  - **متدها**:
    - `to_dict()` و `to_json()` برای تبدیل پیام به دیکشنری یا JSON.
    - `from_dict()` و `from_json()` برای ایجاد نمونه از داده‌های دریافتی.

- **DataResponse**  
  کلاس اصلی پیام پاسخ؛ ترکیبی از `MessageMetadata` و `ResponsePayload`.  
  - **متدها** مشابه DataRequest.

#### ۳.۳ توابع کمکی
- **create_data_request(...)**  
  تابعی جهت ایجاد یک DataRequest با دریافت پارامترهایی نظیر model_id، data_type، data_source، priority، response_topic و غیره.  
  - استخراج مقدار عددی اولویت و رشته‌ای منبع درخواست.
  - ایجاد نمونه‌های MessageMetadata و RequestPayload.
  - بازگشت یک نمونه DataRequest.

- **create_data_response(...)**  
  تابعی جهت ایجاد یک DataResponse با دریافت پارامترهایی نظیر request_id، model_id، status، data، data_type، data_source و غیره.
  - ایجاد نمونه‌های MessageMetadata و ResponsePayload.
  - بازگشت یک نمونه DataResponse.

- **is_valid_data_request(request)**  
  تابعی جهت اعتبارسنجی ساختار DataRequest؛ بررسی وجود فیلدهای ضروری در metadata و payload.

- **is_valid_data_response(response)**  
  تابعی جهت اعتبارسنجی ساختار DataResponse؛ بررسی وجود فیلدهای ضروری در metadata و payload.

---

### 4️⃣ مدیریت موضوعات کافکا (`topic_manager.py`)
این بخش مسئول مدیریت موضوعات (Topics) در سیستم پیام‌رسانی است. ایجاد، اعتبارسنجی، حذف و لیست‌کردن موضوعات توسط کلاس **TopicManager** انجام می‌شود.

#### ۴.۱ Enum دسته‌بندی موضوعات
- **TopicCategory**  
  دسته‌بندی انواع موضوعات برای کاربردهای مختلف:
  - `REQUEST`: درخواست‌ها.
  - `RESPONSE`: پاسخ‌ها.
  - `EVENT`: رویدادها.
  - `METRIC`: متریک‌ها.
  - `LOG`: لاگ‌ها.
  - `INTERNAL`: ارتباطات داخلی.
  - `COMMAND`: دستورات.
  - `NOTIFICATION`: اعلان‌ها.

#### ۴.۲ کلاس TopicManager
- **هدف**:  
  مدیریت موضوعات کافکا برای ماژول‌های سیستم؛ ایجاد ساختار پایه موضوعات و مدیریت موضوعات اختصاصی مدل‌ها.

- **متدهای اصلی**:

  - **`__init__(kafka_service)`**  
    مقداردهی اولیه با دریافت شیء kafka_service جهت استفاده در عملیات‌های مدیریتی.
  
  - **`_initialize_topics()`**  
    تعریف ساختار پایه موضوعات به صورت دیکشنری:
    - **data_requests**: درخواست‌های جمع‌آوری داده از Balance به Data.
    - **data_responses_prefix**: پیشوند نتایج مدل‌ها؛ به صورت پویا برای هر مدل ایجاد می‌شود.
    - **models_requests**: درخواست‌های مدل‌ها به Balance.
    - **balance_metrics**: متریک‌های مربوط به Balance.
    - **balance_events**: رویدادهای مربوط به Balance.
    - **system_logs**: لاگ‌های داخلی سیستم.
    - همچنین نگهداری موضوعات ایجاد شده در مجموعه `created_topics` و موضوعات اختصاصی مدل‌ها در `model_topics`.

  - **`get_topic_name(topic_key)`**  
    دریافت نام کامل موضوع بر اساس کلید داده شده؛ در صورت عدم وجود کلید، همان کلید بازگردانده می‌شود.

  - **`get_model_result_topic(model_id)`**  
    ساخت و ذخیره نام موضوع نتایج برای یک مدل خاص؛ در صورت تکراری بودن از کش استفاده می‌کند.

  - **`ensure_topic_exists(topic_name, partitions, replication)`**  
    اطمینان از وجود یک موضوع؛ در صورت عدم وجود از طریق kafka_service آن را ایجاد می‌کند و در مجموعه created_topics ذخیره می‌کند.

  - **`ensure_model_topic(model_id)`**  
    اطمینان از ایجاد موضوع نتایج برای یک مدل؛ بر اساس پیشوند تعریف‌شده.

  - **`initialize_all_topics()`**  
    ایجاد تمامی موضوعات پایه مورد نیاز سیستم با استفاده از ensure_topic_exists.
  
  - **`list_all_topics()`**  
    دریافت لیست تمام موضوعات موجود در Kafka.
  
  - **`list_model_topics()`**  
    دریافت لیست موضوعات اختصاصی مدل‌ها به صورت تاپل (model_id, topic_name).
  
  - **`delete_topic(topic_name)`**  
    حذف یک موضوع از Kafka؛ به همراه پاکسازی از لیست‌های داخلی.
  
  - **`delete_model_topic(model_id)`**  
    حذف موضوع اختصاصی یک مدل؛ در صورت یافت نشدن، هشدار داده می‌شود.

---

### 5️⃣ صادرسازی API‌های اصلی (`__init__.py`)
فایل __init__.py ماژول Messaging مسئول صادرسازی تمامی کلاس‌ها، توابع و ثابت‌های تعریف‌شده در بخش‌های مختلف این ماژول است.  
با استفاده از این فایل، سایر بخش‌های سیستم می‌توانند به سادگی به امکانات Messaging دسترسی داشته باشند.

- **صادرسازی ثابت‌ها**:  
  تمامی ثابت‌های مربوط به نسخه، پیشوندهای موضوعات، تنظیمات پیش‌فرض، وضعیت‌های پیام و برچسب‌های خطا.

- **صادرسازی ساختارهای پیام**:  
  انواع داده‌ها، کلاس‌های MessageMetadata، RequestPayload، ResponsePayload، DataRequest، DataResponse و توابع کمکی مربوط به آن‌ها.

- **صادرسازی مدیریت موضوعات**:  
  TopicCategory و TopicManager.

- **صادرسازی سرویس کافکا**:  
  KafkaService و نمونه singleton `kafka_service`.

- **لیست نمادهای صادرشده**:  
  __all__ شامل تمامی کلاس‌ها، توابع و ثابت‌هایی است که می‌توانند توسط سایر ماژول‌ها استفاده شوند.

---

## 🔄 تعامل بین اجزا

1. **تعامل بین ثابت‌ها و کلاس‌های پیام**  
   - ثابت‌ها و تنظیمات موجود در `constants.py` به عنوان پایه برای نام‌گذاری موضوعات، تنظیمات تایم‌اوت و اولویت‌ها عمل می‌کنند.
   - کلاس‌های `MessageMetadata`، `RequestPayload` و `ResponsePayload` با استفاده از این ثابت‌ها، پیام‌های استانداردی را تعریف می‌کنند.

2. **تعامل بین KafkaService و TopicManager**  
   - **KafkaService** وظیفه برقراری ارتباط با سرور Kafka، ارسال و دریافت پیام‌ها و مدیریت اشتراک‌ها را بر عهده دارد.
   - **TopicManager** با استفاده از KafkaService، اطمینان حاصل می‌کند که موضوعات مورد نیاز ایجاد شده و به صورت پویا مدیریت می‌شوند.

3. **تعامل بین ماژول‌های ارسال و دریافت پیام**  
   - توابع کمکی در `message_schemas.py` مانند `create_data_request` و `create_data_response` به ایجاد پیام‌های صحیح کمک می‌کنند.
   - توابع اعتبارسنجی (`is_valid_data_request` و `is_valid_data_response`) از بروز خطاهای ساختاری جلوگیری می‌کنند.

---

## 📊 استراتژی‌های بهینه‌سازی و امنیتی

### 1. بهینه‌سازی ارتباطات
- **ارتباط غیرهمزمان**:  
  استفاده از async/await در متدهای KafkaService تضمین می‌کند که پیام‌ها به صورت غیرهمزمان ارسال و دریافت شوند.
- **ارسال دسته‌ای پیام‌ها**:  
  متد `send_batch` بهره‌وری سیستم را در مواقع پردازش حجم بالای داده افزایش می‌دهد.
- **اعتبارسنجی پیام‌ها**:  
  توابع `is_valid_data_request` و `is_valid_data_response` از صحت ساختار پیام‌ها اطمینان حاصل می‌کنند.

### 2. استراتژی‌های امنیتی
- **مدیریت خطا و گزارش‌دهی**:  
  استفاده از log‌های دقیق (با پیام‌های اطلاعاتی و خطا) در تمامی متدها، امکان نظارت و رفع اشکال در زمان واقعی را فراهم می‌کند.
- **کنترل دسترسی به موضوعات**:  
  مدیریت دقیق اشتراک‌ها (subscribe/unsubscribe) و اعتبارسنجی موضوعات از بروز مشکلات احتمالی جلوگیری می‌کند.

