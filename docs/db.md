# ساختار ساده‌شده مدیریت اسکیمای دیتابیس‌ها

با توجه به بازخورد شما، ساختاری ساده‌تر و کاربردی‌تر طراحی کردم که همچنان تمیز باشد ولی پیچیدگی کمتری داشته باشد:

## ساختار پیشنهادی ساده‌شده

```
infrastructure/
└── schemas/
    ├── __init__.py                # نقطه ورود اصلی و تابع راه‌اندازی
    ├── base.py                    # کلاس‌های پایه و اینترفیس‌ها
    │
    ├── timescaledb/
    │   ├── __init__.py            # ثبت و مدیریت جداول TimescaleDB
    │   ├── manager.py             # مدیریت چک کردن و ساختن جداول
    │   └── tables/                # هر جدول در یک فایل جداگانه
    │       ├── __init__.py
    │       ├── schema_version.py
    │       ├── time_series.py
    │       └── users.py
    │
    ├── clickhouse/
    │   ├── __init__.py            # ثبت و مدیریت جداول ClickHouse
    │   ├── manager.py             # مدیریت چک کردن و ساختن جداول
    │   └── tables/                # هر جدول در یک فایل جداگانه
    │       ├── __init__.py
    │       ├── schema_version.py
    │       ├── events.py
    │       └── sessions.py
    │
    └── milvus/
        ├── __init__.py            # ثبت و مدیریت کالکشن‌های Milvus
        ├── manager.py             # مدیریت چک کردن و ساختن کالکشن‌ها
        └── collections/           # هر کالکشن در یک فایل جداگانه
            ├── __init__.py
            ├── schema_version.py
            └── documents.py
```

بدون استفاده از تصاویر، معماری کلی دیتابیس برای سیستم فدراتیو مدل‌های تخصصی (وکیل، پزشک، زبان‌شناس، برنامه‌نویس) به این صورت است:

```
┌─────────────────────────────────────────────────────────────────┐
│                     FEDERATION MANAGEMENT LAYER                 │
├────────────┬─────────────┬────────────────┬───────────────────┤
│  LEGAL AI  │ MEDICAL AI  │ LANGUAGE AI    │ PROGRAMMING AI    │
│  DATABASE  │  DATABASE   │  DATABASE      │  DATABASE         │
├────────────┴─────────────┴────────────────┴───────────────────┤
│                      SHARED KNOWLEDGE LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐    │
│ │   VECTOR    │   │  RELATIONAL │   │     ANALYTICAL      │    │
│ │ COLLECTIONS │   │   TABLES    │   │      TABLES         │    │
│ │  (MILVUS)   │   │(TIMESCALEDB)│   │    (CLICKHOUSE)     │    │
│ └─────────────┘   └─────────────┘   └─────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 💾 ساختار تفصیلی دیتابیس‌ها

### 1️⃣ لایه وکتور (Milvus)

| کالکشن | توضیح | فیلدهای اصلی | کاربرد |
|--------|------|-------------|---------|
| **shared_knowledge_base** | پایگاه دانش مشترک | id, embedding, content, source, category, quality_score | مرکز تبادل دانش بین مدل‌های تخصصی |
| **legal_knowledge** | دانش حقوقی | id, embedding, content, legal_domain, jurisdiction, precedent_ids | مختص مدل وکیل |
| **medical_knowledge** | دانش پزشکی | id, embedding, content, medical_field, evidence_level, peer_reviewed | مختص مدل پزشکی |
| **language_knowledge** | دانش زبان‌شناسی | id, embedding, content, language_code, fluency_level, grammar_rules | مختص مدل زبان‌شناس |
| **programming_knowledge** | دانش برنامه‌نویسی | id, embedding, content, language, framework, complexity_level | مختص مدل برنامه‌نویس |
| **cross_domain_mappings** | ارتباطات بین‌دامنه‌ای | id, source_domain, target_domain, source_id, target_id, confidence | نگاشت مفاهیم بین دامنه‌های مختلف |

### 2️⃣ لایه رابطه‌ای و سری زمانی (TimescaleDB)

| جدول | توضیح | فیلدهای اصلی | کاربرد |
|------|------|-------------|---------|
| **knowledge_relationships** | روابط بین دانش‌ها | id, source_id, target_id, relationship_type, strength, created_at | نگهداری گراف دانش |
| **knowledge_updates** | به‌روزرسانی‌های دانش | id, knowledge_id, previous_version, current_version, updated_at, changes | ردیابی تغییرات دانش |
| **federation_logs** | لاگ تبادلات فدراتیو | id, source_model, target_model, knowledge_id, timestamp, success, details | ثبت تعاملات بین مدل‌ها |
| **model_performance** | عملکرد مدل‌ها | timestamp, model_id, query_type, response_time, accuracy_score, usage_context | سری زمانی عملکرد مدل‌ها |
| **knowledge_quality** | کیفیت دانش تولیدی | id, knowledge_id, reviewer_model, accuracy, relevance, timestamp | ارزیابی کیفیت بین مدل‌ها |

### 3️⃣ لایه تحلیلی (ClickHouse)

| جدول | توضیح | فیلدهای اصلی | کاربرد |
|------|------|-------------|---------|
| **model_interactions** | تعاملات مدل‌ها | event_time, model_id, request_type, domain, tokens_used, latency, success | تحلیل استفاده از مدل‌ها |
| **knowledge_transfer_stats** | آمار انتقال دانش | date, source_domain, target_domain, transfers_count, accepted_count, improved_performance | آمار انتقال دانش بین دامنه‌ها |
| **domain_performance** | عملکرد دامنه‌ای | date, domain, query_count, avg_latency, accuracy, user_satisfaction | تحلیل کارایی هر دامنه تخصصی |
| **cross_domain_usage** | استفاده بین‌دامنه‌ای | date, primary_domain, secondary_domain, query_count, performance_gain | تحلیل همکاری بین دامنه‌ها |
| **knowledge_evolution** | تکامل دانش | date, domain, new_entries, updated_entries, deprecated_entries, quality_trend | تحلیل تکامل پایگاه دانش |

## 🔄 مکانیزم فدراسیون و اشتراک دانش

```
┌──────────────────────────────────────────────────────────────────────┐
│                       FEDERATION MECHANISM                           │
│                                                                      │
│  ┌──────────┐      ┌──────────┐      ┌─────────┐      ┌───────────┐ │
│  │          │◄────►│          │◄────►│         │◄────►│           │ │
│  │ LEGAL AI │      │MEDICAL AI│      │LANG. AI │      │PROGRAM. AI│ │
│  │          │      │          │      │         │      │           │ │
│  └────┬─────┘      └────┬─────┘      └────┬────┘      └─────┬─────┘ │
│       │                 │                 │                 │        │
│       │                 ▼                 │                 │        │
│       │      ┌────────────────────┐      │                 │        │
│       └─────►│                    │◄─────┘                 │        │
│              │  SHARED KNOWLEDGE  │                        │        │
│              │       BASE         │◄───────────────────────┘        │
│              │                    │                                  │
│              └────────────────────┘                                  │
│                        ▲                                             │
│                        │                                             │
│              ┌────────────────────┐                                  │
│              │   KNOWLEDGE        │                                  │
│              │   QUALITY          │                                  │
│              │   EVALUATION       │                                  │
│              └────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────────┘
```

### مکانیزم تبادل دانش بین مدل‌ها

1. **ارسال و دریافت مستقیم:**
   - هر مدل می‌تواند دانش خود را با مدل دیگر به‌طور مستقیم به اشتراک بگذارد
   - مثال: مدل پزشکی می‌تواند اطلاعات درباره مسئولیت قانونی درمان‌ها را از مدل حقوقی درخواست کند

2. **اشتراک‌گذاری از طریق پایگاه دانش مشترک:**
   - هر مدل می‌تواند اطلاعات ارزشمند را در پایگاه دانش مشترک ثبت کند
   - سایر مدل‌ها می‌توانند از این پایگاه دانش استفاده کنند
   - فیلتر کیفیت، صحت اطلاعات را تضمین می‌کند

3. **ارزیابی تبادلی:**
   - مدل‌ها کیفیت اطلاعات سایر مدل‌ها را ارزیابی می‌کنند
   - اطلاعات با کیفیت بالاتر اولویت بیشتری در رتبه‌بندی دارند

## 🚀 مقیاس‌پذیری و کارایی

### استراتژی‌های کلیدی برای مقیاس‌پذیری

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SCALABILITY STRATEGY                         │
│                                                                     │
│  ┌────────────────┐     ┌────────────────┐     ┌─────────────────┐ │
│  │  HORIZONTAL    │     │   VERTICAL     │     │   INTELLIGENT   │ │
│  │  SCALING       │     │   SCALING      │     │     CACHING     │ │
│  └────────────────┘     └────────────────┘     └─────────────────┘ │
│                                                                     │
│  ┌────────────────┐     ┌────────────────┐     ┌─────────────────┐ │
│  │  DOMAIN        │     │   TEMPORAL     │     │   GEOGRAPHICAL  │ │
│  │  SHARDING      │     │   PARTITIONING │     │    DISTRIBUTION │ │
│  └────────────────┘     └────────────────┘     └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

1. **شارد‌بندی دامنه‌ای (Domain Sharding):**
   - داده‌های هر تخصص در شارد جداگانه ذخیره می‌شود
   - مثال: تمام دانش حقوقی در شاردهای اختصاصی حقوقی
   - مزیت: جستجو و بازیابی سریع‌تر در هر دامنه تخصصی

2. **پارتیشن‌بندی زمانی (Temporal Partitioning):**
   - داده‌ها بر اساس زمان پارتیشن‌بندی می‌شوند
   - در TimescaleDB برای جداول سری زمانی
   - مزیت: حذف راحت‌تر داده‌های قدیمی و بهبود عملکرد جستجو

3. **کش هوشمند چندسطحی:**
   - سطح 1: کش درون‌مدلی (نتایج محاسبات داخلی)
   - سطح 2: کش بین‌مدلی (نتایج جستجوهای مکرر بین مدل‌ها)
   - سطح 3: کش پیش‌بینی‌کننده (پیش‌بارگذاری داده‌های احتمالی)

4. **توزیع جغرافیایی:**
   - توزیع فیزیکی داده‌ها بر اساس موقعیت کاربران
   - کاهش تأخیر برای درخواست‌های منطقه‌ای
   - پشتیبانی از فعالیت در شرایط قطع ارتباط موقت

## 🛡️ استراتژی‌های کلیدی برای بهبود عملکرد

### 1. دسته‌بندی داده‌ها بر اساس فرکانس استفاده

| دسته | فرکانس استفاده | نوع ذخیره‌سازی | مثال |
|------|----------------|---------------|------|
| **داغ** | بسیار زیاد | کش در RAM | مفاهیم پایه در هر تخصص |
| **گرم** | متوسط | SSD سریع | دانش کاربردی روزانه |
| **سرد** | کم | ذخیره‌سازی استاندارد | دانش تخصصی عمیق |
| **منجمد** | بسیار کم | ذخیره‌سازی ارزان | داده‌های آرشیوی |

### 2. استراتژی‌های ویژه برای هر دیتابیس

| دیتابیس | استراتژی کلیدی | مزیت |
|---------|----------------|------|
| **Milvus** | ایندکس HNSW برای جستجوی ANN | جستجوی معنایی بسیار سریع |
| | پارتیشن‌بندی براساس موضوع | جستجو در فضای محدودتر |
| | لود دینامیک کالکشن‌ها | مصرف حافظه بهینه |
| **TimescaleDB** | چانک‌های بهینه‌شده | کوئری‌های سریع در بازه‌های زمانی |
| | ایندکس‌های ترکیبی | بهبود عملکرد کوئری‌های پیچیده |
| | متریالایزد ویوهای تجمیعی | سرعت بالا در گزارش‌گیری |
| **ClickHouse** | پارتیشن‌بندی زمانی | پرس‌وجوهای تاریخی سریع |
| | فشرده‌سازی ستونی | صرفه‌جویی در فضا و افزایش سرعت |
| | موتور MergeTree | عملکرد بالا در تحلیل‌های آماری |

### 3. یکپارچه‌سازی و ارتباط چندسطحی

| سطح ارتباط | شکل دسترسی | کاربرد |
|------------|------------|---------|
| **درون مدلی** | مستقیم به پایگاه داده | جستجوهای تخصصی عمیق |
| **بین مدلی مستقیم** | API یکپارچه | تبادل دانش بین دو مدل خاص |
| **بین مدلی عمومی** | دسترسی به پایگاه دانش مشترک | جستجوی دانش عمومی |
| **فدراسیون کامل** | دسترسی به تمام منابع | تحلیل‌های چندبعدی و پیچیده |



# مستند ساختار پوشه Storage در پروژه Smart Whale

## 1. معرفی و هدف

پوشه `storage` در پروژه Smart Whale با هدف مدیریت متمرکز، مقیاس‌پذیر و کارآمد داده‌های پایدار سیستم طراحی شده است. این ساختار امکان سازماندهی داده‌های مختلف (دیتابیس‌ها، کش‌ها، لاگ‌ها و فایل‌ها) را به صورت منسجم و منظم فراهم می‌کند و اجازه می‌دهد تا مدل‌های مختلف داده‌های خود را به صورت جدا از یکدیگر مدیریت کنند.

مزایای اصلی این ساختار:
- **مدیریت متمرکز داده‌های پایدار**
- **تفکیک داده‌های هر مدل** (زبانی، برنامه‌نویس و غیره)
- **پشتیبان‌گیری و بازیابی ساده‌تر**
- **مدیریت و نظارت بهتر بر فضای دیسک**
- **مقیاس‌پذیری و امکان توسعه آسان‌تر**

## 2. ساختار کلی پوشه Storage

```
storage/
├── models/                 # داده‌های تفکیک شده مدل‌ها
│   ├── language/           # مدل زبانی
│   │   ├── db/             # داده‌های دیتابیس مدل زبانی
│   │   │   ├── schemas/    # اسکیماهای دیتابیس
│   │   │   │   ├── timescaledb/
│   │   │   │   ├── clickhouse/
│   │   │   │   └── milvus/
│   │   ├── cache/          # کش مدل زبانی
│   │   └── uploads/        # فایل‌های آپلود شده مرتبط با مدل زبانی
│   ├── developer/          # مدل برنامه‌نویس
│   │   ├── db/
│   │   ├── cache/
│   │   └── uploads/
│   └── other_models/       # سایر مدل‌ها
│
├── shared/                 # داده‌های مشترک بین مدل‌ها
│   ├── db/                 # داده‌های دیتابیس مشترک
│   │   ├── timescaledb/
│   │   ├── clickhouse/
│   │   └── milvus/
│   ├── cache/              # داده‌های کش مشترک
│   │   └── redis/
│   ├── uploads/            # فایل‌های آپلودی مشترک
│   ├── kafka/              # داده‌های Kafka
│   │   ├── data/
│   │   └── zookeeper/
│   └── tmp/                # فایل‌های موقت
│
├── logs/                   # لاگ‌های سیستم
│   ├── app/
│   ├── access/
│   ├── metrics/
│   └── errors/
│
├── backups/                # پشتیبان‌گیری
│   ├── language/
│   ├── developer/
│   └── shared/
│
├── monitoring/             # داده‌های مانیتورینگ
│   ├── prometheus/
│   └── grafana/
│
├── config/                 # فایل‌های پیکربندی
│   ├── clickhouse/
│   ├── grafana/
│   │   └── provisioning/
│   └── prometheus/
│
└── scripts/                # اسکریپت‌های مدیریتی
    ├── __init__.py
    ├── base_schema.py
    ├── db_manager.py
    ├── init_schema.py
    ├── setup.py
    └── setup_env.ps1
```

## 3. جزئیات پوشه Scripts

پوشه `scripts` حاوی اسکریپت‌های پایتون و PowerShell برای مدیریت اسکیماهای دیتابیس است. این اسکریپت‌ها به ما امکان می‌دهند تا اسکیماهای دیتابیس را به صورت کد تعریف کنیم، به صورت خودکار آن‌ها را کشف کنیم و در دیتابیس‌های مختلف ایجاد کنیم.

### 3.1. فایل base_schema.py

**هدف**: تعریف کلاس‌های پایه برای اسکیماهای دیتابیس.

**محتوا**:
- کلاس پایه `SchemaObject`: کلاس انتزاعی برای تمامی انواع اسکیما
- کلاس `TimescaleDBSchema`: کلاس پایه برای اسکیمای TimescaleDB (PostgreSQL)
- کلاس `ClickHouseSchema`: کلاس پایه برای اسکیمای ClickHouse
- کلاس `MilvusSchema`: کلاس پایه برای اسکیمای Milvus (پایگاه داده برداری)

**نحوه استفاده**: این کلاس‌ها به عنوان کلاس پایه برای تعریف اسکیماهای مختلف استفاده می‌شوند. هر اسکیما باید یکی از کلاس‌های فرزند را گسترش دهد و متدهای مجرد آن را پیاده‌سازی کند.

### 3.2. فایل db_manager.py

**هدف**: مدیریت اسکیماهای دیتابیس، کشف خودکار آن‌ها و ایجاد آن‌ها در دیتابیس‌های مختلف.

**محتوا**:
- کلاس `DatabaseManager`: مدیریت کننده اسکیماهای دیتابیس
  - متد `discover_schemas`: کشف خودکار اسکیماها از پوشه‌های مختلف
  - متد `create_all_schemas`: ایجاد تمامی اسکیماها در دیتابیس‌ها
  - متد `create_schema`: ایجاد یک اسکیما خاص در دیتابیس مربوطه
  - متدهای مرتبط با هر دیتابیس خاص: `_create_timescaledb_schema`, `_create_clickhouse_schema`, `_create_milvus_schema`
  - متد `get_schema_info`: دریافت اطلاعات اسکیماها
  - متد `save_schema_info`: ذخیره اطلاعات اسکیماها در فایل JSON

**نحوه استفاده**: این کلاس برای کشف اسکیماهای تعریف شده در کد و ایجاد آن‌ها در دیتابیس‌های مختلف استفاده می‌شود.

### 3.3. فایل init_schema.py

**هدف**: اسکریپت اصلی برای اجرا و ایجاد اسکیماهای دیتابیس.

**محتوا**:
- تابع `main`: تابع اصلی برنامه
- توابع کمکی برای اتصال به دیتابیس‌ها: `initialize_timescaledb`, `initialize_clickhouse`, `initialize_milvus`
- پارسینگ آرگومان‌های خط فرمان برای مشخص کردن اسکیماهایی که باید ایجاد شوند یا نمایش داده شوند

**دستور اجرا**:
```bash
# برای کشف و نمایش اسکیماها (بدون ایجاد)
python storage/scripts/init_schema.py --info

# برای ایجاد تمام اسکیماها
python storage/scripts/init_schema.py --all

# برای ایجاد اسکیماهای مدل خاص
python storage/scripts/init_schema.py --model shared

# برای ایجاد اسکیماها در دیتابیس خاص
python storage/scripts/init_schema.py --timescaledb
```

### 3.4. فایل setup.py

**هدف**: اسکریپت راه‌اندازی پوشه‌های مورد نیاز برای اسکیماهای دیتابیس.

**محتوا**:
- تابع `main`: تابع اصلی برنامه
- توابع کمکی برای ایجاد پوشه‌ها و فایل‌های `__init__.py`

**دستور اجرا**:
```bash
# برای راه‌اندازی پوشه‌های مدل‌های پیش‌فرض
python storage/scripts/setup.py

# برای راه‌اندازی پوشه‌های مدل‌های خاص
python storage/scripts/setup.py --models language developer

# برای راه‌اندازی پوشه‌های با دیتابیس‌های خاص
python storage/scripts/setup.py --db-types timescaledb clickhouse
```

### 3.5. فایل setup_env.ps1

**هدف**: اسکریپت PowerShell برای راه‌اندازی محیط، شامل ایجاد پوشه‌های مورد نیاز.

**محتوا**:
- توابع کمکی: `Create-Directory`, `Create-InitFile`
- ایجاد پوشه‌های مختلف برای مدل‌ها، دیتابیس‌ها و ...
- ایجاد فایل‌های `__init__.py` در هر پوشه

**دستور اجرا**:
```powershell
# اجرای اسکریپت PowerShell
& "storage\scripts\setup_env.ps1"
```

**نکته**: این اسکریپت باید در مسیر اصلی پروژه اجرا شود.

## 4. جزئیات پوشه Shared

پوشه `shared` حاوی داده‌های مشترک بین تمام مدل‌هاست. در بخش `db/schemas` اسکیماهای مشترک قرار دارند.

### 4.1. اسکیماهای TimescaleDB

#### 4.1.1. اسکیمای schema_version.py

**هدف**: نگهداری نسخه‌های اسکیماهای ایجاد شده و تاریخچه تغییرات.

**کلاس**: `SchemaVersionTable`

**جدول و فیلدها**:
```sql
CREATE TABLE IF NOT EXISTS public.schema_version (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL UNIQUE,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### 4.1.2. اسکیمای users.py

**هدف**: نگهداری اطلاعات کاربران سیستم.

**کلاس**: `UsersTable`

**جدول و فیلدها**:
```sql
CREATE TABLE IF NOT EXISTS public.users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    profile_picture VARCHAR(255),
    bio TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE
);
```

#### 4.1.3. اسکیمای chats.py

**هدف**: نگهداری چت‌های کاربران و پیام‌های چت.

**کلاس**: `ChatsTable` و `ChatMessagesTable`

**جداول و فیلدها**:
```sql
-- جدول چت‌ها
CREATE TABLE IF NOT EXISTS public.chats (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_archived BOOLEAN DEFAULT FALSE,
    settings JSONB
);

-- جدول پیام‌های چت
CREATE TABLE IF NOT EXISTS public.chat_messages (
    id SERIAL PRIMARY KEY,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tokens_used INTEGER,
    metadata JSONB
);
```

### 4.2. اسکیماهای ClickHouse

#### 4.2.1. اسکیمای events.py

**هدف**: نگهداری رویدادهای سیستم برای تحلیل و آمارگیری.

**کلاس**: `EventsTable`

**جدول و فیلدها**:
```sql
CREATE TABLE IF NOT EXISTS default.events (
    event_date Date,
    event_time DateTime,
    event_type String,
    user_id UInt64,
    session_id String,
    properties String,  -- JSON string
    ip_address String,
    user_agent String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (event_type, event_time);
```

#### 4.2.2. اسکیمای usage_stats.py

**هدف**: نگهداری آمار استفاده کاربران از سیستم.

**کلاس**: `UsageStatsTable`

**جدول و فیلدها**:
```sql
CREATE TABLE IF NOT EXISTS default.usage_stats (
    date Date,
    user_id UInt64,
    model_name String,
    tokens_input UInt32,
    tokens_output UInt32,
    processing_time_ms UInt32,
    request_count UInt32,
    average_latency_ms Float32,
    error_count UInt16
) ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id, model_name);
```

### 4.3. اسکیماهای Milvus

#### 4.3.1. اسکیمای knowledge_base.py

**هدف**: پایگاه دانش مشترک برای همه مدل‌ها.

**کلاس**: `KnowledgeBaseCollection`

**فیلدها**:
```python
fields = [
    {"name": "id", "type": DataType.INT64, "is_primary": True},
    {"name": "embedding", "type": DataType.FLOAT_VECTOR, "dim": 1536},
    {"name": "content", "type": DataType.VARCHAR, "max_length": 4096},
    {"name": "source", "type": DataType.VARCHAR, "max_length": 255},
    {"name": "category", "type": DataType.VARCHAR, "max_length": 100},
    {"name": "created_at", "type": DataType.INT64},
    {"name": "metadata", "type": DataType.JSON}
]
```

## 5. استراتژی پیشنهادی برای ماژول Language

با توجه به مطالعه فایل‌های ارائه شده، برای ماژول زبانی (language) استراتژی زیر پیشنهاد می‌شود:

### 5.1. ساختار پوشه‌بندی

```
storage/models/language/db/schemas/
├── timescaledb/
│   ├── __init__.py              # وارد کردن تمام ماژول‌ها
│   ├── dialects/
│   │   ├── __init__.py          # لیست تمام جداول dialects
│   │   ├── dialects_table.py
│   │   ├── dialect_features_table.py
│   │   ├── dialect_words_table.py
│   │   ├── dialect_conversion_rules_table.py
│   │   ├── dialect_detection_history_table.py
│   │   └── dialect_text_vectors_table.py
│   ├── domain/
│   │   ├── __init__.py          # لیست تمام جداول domain
│   │   ├── domains_table.py
│   │   ├── domain_concepts_table.py
│   │   ├── concept_relations_table.py
│   │   └── concept_attributes_table.py
│   ├── contextual/
│   │   ├── __init__.py          # لیست تمام جداول contextual
│   │   ├── conversations_table.py
│   │   ├── messages_table.py
│   │   ├── user_intents_table.py
│   │   └── context_knowledge_table.py
│   └── utils/                   # جداول مشترک یا سرویس‌های عمومی
│       ├── __init__.py
│       └── schema_version_table.py
├── clickhouse/
│   └── ...                      # ساختار مشابه برای ClickHouse
└── milvus/
    └── ...                      # ساختار مشابه برای Milvus
```

### 5.2. نام‌گذاری استاندارد

- **نام فایل‌ها**: `[table_name].py` یا `[table_name]_table.py`
- **نام کلاس‌ها**: `[TableName]Schema`

### 5.3. پیاده‌سازی جداول ماژول Dialects

بر اساس فایل `data_access.py` در ماژول `dialects` که ارائه شده است، جداول زیر باید پیاده‌سازی شوند:

1. **جدول dialects**:
```sql
CREATE TABLE IF NOT EXISTS dialects (
    dialect_id String,
    dialect_name String,
    dialect_code String,
    region String,
    description String,
    parent_dialect String DEFAULT '',
    popularity Float32 DEFAULT 0,
    source String,
    discovery_time DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (dialect_id, discovery_time)
```

2. **جدول dialect_features**:
```sql
CREATE TABLE IF NOT EXISTS dialect_features (
    feature_id String,
    dialect_id String,
    feature_type String,
    feature_pattern String,
    description String,
    examples Array(String),
    confidence Float32,
    source String,
    usage_count UInt32 DEFAULT 1,
    discovery_time DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (feature_id, dialect_id, discovery_time)
```

3. **جدول dialect_words**:
```sql
CREATE TABLE IF NOT EXISTS dialect_words (
    word_id String,
    dialect_id String,
    word String,
    standard_equivalent String,
    definition String,
    part_of_speech String DEFAULT '',
    usage Array(String),
    confidence Float32,
    source String,
    usage_count UInt32 DEFAULT 1,
    discovery_time DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (word_id, dialect_id, discovery_time)
```

4. **جدول dialect_conversion_rules**:
```sql
CREATE TABLE IF NOT EXISTS dialect_conversion_rules (
    rule_id String,
    source_dialect String,
    target_dialect String,
    rule_type String,
    rule_pattern String,
    replacement String,
    description String,
    examples Array(String),
    confidence Float32,
    source String,
    usage_count UInt32 DEFAULT 1,
    discovery_time DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (rule_id, source_dialect, target_dialect, discovery_time)
```

5. **جدول dialect_detection_history**:
```sql
CREATE TABLE IF NOT EXISTS dialect_detection_history (
    detection_id String,
    text String,
    detected_dialect String,
    confidence Float32,
    dialect_features Array(String),
    detection_time DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (detection_id, detection_time)
```

6. **جدول dialect_text_vectors**:
```sql
CREATE TABLE IF NOT EXISTS dialect_text_vectors (
    text_hash String,
    text String,
    dialect_id String,
    vector Array(Float32),
    timestamp DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (text_hash, dialect_id, timestamp)
```

### 5.4. پیاده‌سازی جداول ماژول Domain

بر اساس فایل `domain_data.py` که ارائه شده است، جداول زیر باید پیاده‌سازی شوند:

1. **جدول domains**:
```sql
CREATE TABLE IF NOT EXISTS domains (
    domain_id String,
    domain_name String,
    domain_code String,
    parent_domain String,
    description String,
    popularity Float32,
    source String,
    discovery_time DateTime
) ENGINE = MergeTree()
ORDER BY (domain_id, discovery_time)
```

2. **جدول domain_concepts**:
```sql
CREATE TABLE IF NOT EXISTS domain_concepts (
    concept_id String,
    domain_id String,
    concept_name String,
    definition String,
    examples Array(String),
    confidence Float32,
    source String,
    usage_count UInt32,
    discovery_time DateTime
) ENGINE = MergeTree()
ORDER BY (concept_id, domain_id, discovery_time)
```

3. **جدول concept_relations**:
```sql
CREATE TABLE IF NOT EXISTS concept_relations (
    relation_id String,
    source_concept_id String,
    target_concept_id String,
    relation_type String,
    description String,
    confidence Float32,
    source String,
    usage_count UInt32,
    discovery_time DateTime
) ENGINE = MergeTree()
ORDER BY (relation_id, source_concept_id, target_concept_id, discovery_time)
```

4. **جدول concept_attributes**:
```sql
CREATE TABLE IF NOT EXISTS concept_attributes (
    attribute_id String,
    concept_id String,
    attribute_name String,
    attribute_value String,
    description String,
    confidence Float32,
    source String,
    usage_count UInt32,
    discovery_time DateTime
) ENGINE = MergeTree()
ORDER BY (attribute_id, concept_id, discovery_time)
```

### 5.5. پیاده‌سازی جداول ماژول Contextual

بر اساس داکیومنت `ai_models_language_apaptors_persian_language_processors.md` در بخش ماژول Contextual، جداول زیر می‌توانند برای ماژول Contextual پیاده‌سازی شوند:

1. **جدول conversations**:
```sql
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id String,
    title String,
    user_id String,
    type String,
    start_time DateTime,
    end_time DateTime,
    message_count UInt32,
    metadata String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (conversation_id, created_at)
```

2. **جدول messages**:
```sql
CREATE TABLE IF NOT EXISTS messages (
    message_id String,
    conversation_id String,
    role String,
    content String,
    importance Float32,
    created_at DateTime,
    metadata String
) ENGINE = MergeTree()
ORDER BY (message_id, conversation_id, created_at)
```

3. **جدول user_intents**:
```sql
CREATE TABLE IF NOT EXISTS user_intents (
    intent_id String,
    message_id String,
    conversation_id String,
    intent_type String,
    topics Array(String),
    urgency UInt8,
    emotion String,
    context_dependency Float32,
    created_at DateTime
) ENGINE = MergeTree()
ORDER BY (intent_id, message_id, created_at)
```

4. **جدول context_knowledge**:
```sql
CREATE TABLE IF NOT EXISTS context_knowledge (
    knowledge_id String,
    conversation_id String,
    message_id String,
    knowledge_type String,
    content String,
    confidence Float32,
    source String,
    created_at DateTime
) ENGINE = MergeTree()
ORDER BY (knowledge_id, conversation_id, created_at)
```

### 5.6. مراحل پیاده‌سازی

1. **ایجاد ساختار پوشه‌بندی اصلی**
2. **پیاده‌سازی اسکیماها برای ماژول dialects**
3. **پیاده‌سازی اسکیماها برای ماژول domain**
4. **پیاده‌سازی اسکیماها برای ماژول contextual**
5. **ایجاد فایل‌های `__init__.py` برای جمع‌آوری و مدیریت اسکیماها**

با این استراتژی، هر ماژول می‌تواند مستقل از سایر ماژول‌ها توسعه یابد و افزودن جدول جدید فقط شامل ایجاد یک فایل جدید و به‌روزرسانی `__init__.py` مربوطه خواهد بود.