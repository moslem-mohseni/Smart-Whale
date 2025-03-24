# داکیومنت پیاده‌سازی ساختار پوشه `storage`

```
storage/
├── db/                      # فایل‌های دیتابیس
│   ├── timescaledb/         # داده‌های TimescaleDB
│   ├── clickhouse/          # داده‌های ClickHouse
│   └── milvus/              # داده‌های Milvus
├── logs/                    # لاگ‌های سیستم
│   ├── app/                 # لاگ‌های اصلی اپلیکیشن
│   ├── access/              # لاگ‌های دسترسی
│   ├── metrics/             # لاگ‌های متریک‌ها
│   └── errors/              # لاگ‌های خطا
├── cache/                   # داده‌های کش
│   └── redis/               # داده‌های Redis
├── kafka/                   # داده‌های Kafka
├── uploads/                 # فایل‌های آپلود شده
├── backups/                 # پشتیبان‌گیری
│   ├── timescaledb/
│   ├── clickhouse/
│   └── milvus/
└── tmp/                     # فایل‌های موقت
```

## 🎯 هدف

ساختار پوشه `storage` به منظور سازماندهی و مدیریت داده‌های پایدار (دیتابیس‌ها، لاگ‌ها، کش‌ها، و فایل‌ها) در پروژه Smart Whale طراحی شده است. این ساختار امکان مدیریت متمرکز، پشتیبان‌گیری آسان، و نگهداری مقیاس‌پذیر داده‌ها را فراهم می‌کند.

## 📋 پیاده‌سازی ساختار پوشه

### 1. ایجاد ساختار اولیه

ابتدا ساختار اصلی پوشه‌ها را در ریشه پروژه ایجاد کنید:

```bash
mkdir -p storage/{db/{timescaledb,clickhouse,milvus},logs/{app,access,metrics,errors},cache/redis,kafka,uploads,backups/{timescaledb,clickhouse,milvus},tmp}
```

### 2. تنظیم حق دسترسی‌ها

```bash
# تنظیم حق دسترسی برای پوشه‌های دیتابیس
chmod -R 755 storage/db
chmod -R 700 storage/db/timescaledb
chmod -R 700 storage/db/clickhouse

# تنظیم حق دسترسی برای پوشه‌های لاگ
chmod -R 755 storage/logs

# تنظیم حق دسترسی برای پوشه‌های آپلود و موقت
chmod -R 755 storage/uploads
chmod -R 755 storage/tmp
```

## 🔧 پیکربندی سرویس‌ها در `docker-compose.yml`

برای استفاده از این ساختار پوشه در سرویس‌های Docker، باید volumes را در فایل `docker-compose.yml` به‌روزرسانی کنید:

```yaml
services:
  app:
    # ...
    volumes:
      - ./app:/app:ro
      - ./storage/logs/app:/app/logs
      - ./storage/tmp:/app/tmp
      - ./storage/uploads:/app/uploads

  timescaledb:
    # ...
    volumes:
      - ./storage/db/timescaledb:/var/lib/postgresql/data
      
  redis:
    # ...
    volumes:
      - ./storage/cache/redis:/data
      
  zookeeper:
    # ...
    volumes:
      - ./storage/kafka/zookeeper:/var/lib/zookeeper/data
      
  kafka:
    # ...
    volumes:
      - ./storage/kafka/data:/var/lib/kafka/data
      
  clickhouse:
    # ...
    volumes:
      - ./storage/db/clickhouse:/var/lib/clickhouse
      - ./clickhouse/config:/etc/clickhouse-server/config.d
      
  prometheus:
    # ...
    volumes:
      - ./infrastructure/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./storage/logs/metrics/prometheus:/prometheus
      
  grafana:
    # ...
    volumes:
      - ./storage/logs/metrics/grafana:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
```

## 📝 پیکربندی سرویس‌ها

### 1. TimescaleDB

تنظیم مسیر داده‌های TimescaleDB:

```yaml
# در docker-compose.yml
timescaledb:
  environment:
    - PGDATA=/var/lib/postgresql/data
  volumes:
    - ./storage/db/timescaledb:/var/lib/postgresql/data
```

### 2. ClickHouse

تنظیم مسیر داده‌های ClickHouse:

```yaml
# در docker-compose.yml
clickhouse:
  volumes:
    - ./storage/db/clickhouse:/var/lib/clickhouse
    - ./clickhouse/config:/etc/clickhouse-server/config.d
```

در `clickhouse/config/config.xml` اضافه کنید:

```xml
<path>/var/lib/clickhouse</path>
<tmp_path>/var/lib/clickhouse/tmp</tmp_path>
<user_files_path>/var/lib/clickhouse/user_files</user_files_path>
<format_schema_path>/var/lib/clickhouse/format_schemas</format_schema_path>
```

### 3. Milvus

تنظیم مسیر داده‌های Milvus:

```yaml
# در docker-compose.yml
milvus:
  volumes:
    - ./storage/db/milvus:/var/lib/milvus
  environment:
    - MILVUS_DATA_PATH=/var/lib/milvus
```

### 4. Redis

تنظیم مسیر داده‌های Redis:

```yaml
# در docker-compose.yml
redis:
  command: redis-server --appendonly yes --dir /data
  volumes:
    - ./storage/cache/redis:/data
```

### 5. Kafka & Zookeeper

تنظیم مسیر داده‌های Kafka و Zookeeper:

```yaml
# در docker-compose.yml
zookeeper:
  environment:
    - ZOO_DATA_DIR=/var/lib/zookeeper/data
  volumes:
    - ./storage/kafka/zookeeper:/var/lib/zookeeper/data

kafka:
  environment:
    - KAFKA_LOG_DIRS=/var/lib/kafka/data
  volumes:
    - ./storage/kafka/data:/var/lib/kafka/data
```

## 📊 پیکربندی لاگینگ

### 1. پیکربندی لاگینگ اپلیکیشن

در فایل `.env` اضافه کنید:

```
LOG_DIR=./storage/logs/app
ERROR_LOG_DIR=./storage/logs/errors
ACCESS_LOG_DIR=./storage/logs/access
```

در کد اپلیکیشن، از این متغیرها استفاده کنید:

```python
# در بخش تنظیمات لاگینگ اپلیکیشن
import os
import logging
from logging.handlers import RotatingFileHandler

log_dir = os.getenv("LOG_DIR", "./storage/logs/app")
error_log_dir = os.getenv("ERROR_LOG_DIR", "./storage/logs/errors")

# ساخت پوشه‌ها اگر وجود ندارند
os.makedirs(log_dir, exist_ok=True)
os.makedirs(error_log_dir, exist_ok=True)

# تنظیم لاگر اصلی
app_handler = RotatingFileHandler(
    f"{log_dir}/app.log",
    maxBytes=10485760,  # 10MB
    backupCount=10
)

# تنظیم لاگر خطاها
error_handler = RotatingFileHandler(
    f"{error_log_dir}/error.log",
    maxBytes=10485760,  # 10MB
    backupCount=10
)
error_handler.setLevel(logging.ERROR)

# تنظیم لاگر اصلی
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
logger.addHandler(app_handler)
logger.addHandler(error_handler)
```

### 2. پیکربندی لاگینگ دسترسی‌ها

```python
# برای FastAPI
from fastapi import FastAPI, Request
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import os
import logging
from logging.handlers import RotatingFileHandler

access_log_dir = os.getenv("ACCESS_LOG_DIR", "./storage/logs/access")
os.makedirs(access_log_dir, exist_ok=True)

# تنظیم لاگر دسترسی‌ها
access_logger = logging.getLogger("access")
access_logger.setLevel(logging.INFO)
access_handler = RotatingFileHandler(
    f"{access_log_dir}/access.log",
    maxBytes=10485760,  # 10MB
    backupCount=10
)
access_logger.addHandler(access_handler)

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        access_logger.info(
            f"{request.client.host} - {request.method} {request.url.path} {response.status_code} {process_time:.4f}s"
        )
        
        return response

app = FastAPI()
app.add_middleware(AccessLogMiddleware)
```

## 📁 پیکربندی آپلود فایل‌ها

### 1. تنظیم مسیر آپلود فایل‌ها

در فایل `.env` اضافه کنید:

```
UPLOAD_DIR=./storage/uploads
TEMP_DIR=./storage/tmp
```

### 2. استفاده در کد اپلیکیشن

```python
import os
from fastapi import UploadFile, File

upload_dir = os.getenv("UPLOAD_DIR", "./storage/uploads")
temp_dir = os.getenv("TEMP_DIR", "./storage/tmp")

# ساخت پوشه‌ها اگر وجود ندارند
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # ذخیره فایل
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {"filename": file.filename}
```

## 🔄 پیکربندی پشتیبان‌گیری

### 1. افزودن اسکریپت پشتیبان‌گیری

ایجاد فایل `scripts/backup.sh`:

```bash
#!/bin/bash

# تنظیم تاریخ
DATE=$(date +%Y-%m-%d-%H-%M)

# دریافت محیط
ENV=$1
if [ -z "$ENV" ]; then
  ENV="development"
fi

# تنظیم مسیرها
BACKUP_ROOT="./storage/backups"
TIMESCALEDB_BACKUP_DIR="$BACKUP_ROOT/timescaledb"
CLICKHOUSE_BACKUP_DIR="$BACKUP_ROOT/clickhouse"
MILVUS_BACKUP_DIR="$BACKUP_ROOT/milvus"

# ساخت پوشه‌ها
mkdir -p $TIMESCALEDB_BACKUP_DIR
mkdir -p $CLICKHOUSE_BACKUP_DIR
mkdir -p $MILVUS_BACKUP_DIR

# پشتیبان‌گیری از TimescaleDB
echo "Backing up TimescaleDB..."
docker exec smart_whale-timescaledb-1 pg_dump -U user aidb > "$TIMESCALEDB_BACKUP_DIR/timescaledb-$ENV-$DATE.sql"
gzip "$TIMESCALEDB_BACKUP_DIR/timescaledb-$ENV-$DATE.sql"

# پشتیبان‌گیری از ClickHouse
echo "Backing up ClickHouse..."
docker exec smart_whale-clickhouse-1 clickhouse-client --query="BACKUP DATABASE default TO '/var/lib/clickhouse/backups/backup-$ENV-$DATE'"
docker cp smart_whale-clickhouse-1:/var/lib/clickhouse/backups/backup-$ENV-$DATE "$CLICKHOUSE_BACKUP_DIR/"

# پشتیبان‌گیری از Milvus
echo "Backing up Milvus..."
# اینجا باید از API Milvus برای پشتیبان‌گیری استفاده کنید

echo "Backup completed: $DATE"
```

### 2. افزودن پشتیبان‌گیری به crontab

برای اجرای خودکار پشتیبان‌گیری، می‌توانید از crontab استفاده کنید:

```bash
# ویرایش crontab
crontab -e

# افزودن دستور (پشتیبان‌گیری روزانه در ساعت 2 صبح)
0 2 * * * cd /path/to/project && ./scripts/backup.sh production >> ./storage/logs/app/backup.log 2>&1
```

## 🧹 پاکسازی خودکار

### 1. افزودن اسکریپت پاکسازی

ایجاد فایل `scripts/cleanup.sh`:

```bash
#!/bin/bash

# پاکسازی فایل‌های موقت قدیمی‌تر از 7 روز
find ./storage/tmp -type f -mtime +7 -delete

# پاکسازی پشتیبان‌های قدیمی‌تر از 30 روز
find ./storage/backups -type f -mtime +30 -delete

# پاکسازی لاگ‌های قدیمی‌تر از 90 روز
find ./storage/logs -type f -mtime +90 -delete

echo "Cleanup completed: $(date)"
```

### 2. افزودن پاکسازی به crontab

```bash
# ویرایش crontab
crontab -e

# افزودن دستور (پاکسازی هفتگی در روز یکشنبه ساعت 3 صبح)
0 3 * * 0 cd /path/to/project && ./scripts/cleanup.sh >> ./storage/logs/app/cleanup.log 2>&1
```

## 🔍 نظارت بر فضای ذخیره‌سازی

### 1. افزودن اسکریپت نظارت

ایجاد فایل `scripts/monitor_storage.sh`:

```bash
#!/bin/bash

# تنظیم آستانه هشدار (80%)
THRESHOLD=80

# بررسی فضای استفاده شده
USAGE=$(df -h ./storage | awk 'NR==2 {print $5}' | sed 's/%//')

# ارسال هشدار اگر استفاده بیش از آستانه است
if [ $USAGE -gt $THRESHOLD ]; then
  echo "WARNING: Storage usage is at ${USAGE}%, exceeding threshold of ${THRESHOLD}%" | tee -a ./storage/logs/app/storage_alert.log
  
  # اینجا می‌توانید کد ارسال ایمیل یا اعلان را اضافه کنید
fi

# گزارش استفاده از فضا به تفکیک پوشه‌ها
echo "Storage usage report: $(date)" >> ./storage/logs/app/storage_report.log
du -sh ./storage/* | sort -hr >> ./storage/logs/app/storage_report.log
```

### 2. افزودن نظارت به crontab

```bash
# ویرایش crontab
crontab -e

# افزودن دستور (بررسی روزانه ساعت 7 صبح)
0 7 * * * cd /path/to/project && ./scripts/monitor_storage.sh
```

## ✨ مزایای استفاده از این ساختار

1. **انسجام و استاندارد**: همه داده‌های پایدار در یک مکان مرکزی مدیریت می‌شوند
2. **مقیاس‌پذیری**: جداسازی انواع داده‌ها برای مدیریت بهتر حجم
3. **امنیت**: تنظیم دقیق حق دسترسی‌ها برای هر نوع داده
4. **پشتیبان‌گیری آسان**: ساختار منظم برای پشتیبان‌گیری و بازیابی
5. **مدیریت چرخه عمر**: پاکسازی خودکار داده‌های قدیمی
6. **نظارت متمرکز**: امکان نظارت بر رشد و استفاده از فضا
7. **توسعه‌پذیری**: افزودن آسان انواع جدید داده‌ها و سرویس‌ها

با پیاده‌سازی این ساختار، مدیریت داده‌های پایدار در پروژه Smart Whale به شکل قابل‌توجهی بهبود می‌یابد و مقیاس‌پذیری پروژه تضمین می‌شود.