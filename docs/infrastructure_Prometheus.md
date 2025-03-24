# راهنمای پیاده‌سازی ماژول Prometheus در پروژه Smart Whale

## 1. مقدمه
در این راهنما، ماژول **Prometheus** را مشابه **Kafka، Redis و ClickHouse** طراحی و پیاده‌سازی می‌کنیم. هدف این است که **Prometheus** به‌صورت ماژولار و مستقل در **`infrastructure/`** قرار گیرد و سایر ماژول‌ها تنها از سرویس‌های آن استفاده کنند.

---

## 2. ساختار پیشنهادی ماژول `Prometheus`
```plaintext
smart_whale/
│── infrastructure/
│   ├── clickhouse/          # ماژول ClickHouse
│   ├── kafka/               # ماژول Kafka
│   ├── redis/               # ماژول Redis
│   ├── prometheus/          # ماژول Prometheus (جدید)
│   │   ├── adapters/        # آداپتورهای Prometheus
│   │   │   ├── prometheus_client_adapter.py  
│   │   ├── config/          # تنظیمات Prometheus
│   │   │   ├── prometheus_config.py
│   │   │   ├── prometheus.yml
│   │   ├── monitoring/      # متریک‌ها و Exporter
│   │   │   ├── prometheus_exporter.py
│   │   │   ├── __init__.py
│   │   ├── service/         # سرویس‌های Prometheus
│   │   │   ├── metrics_service.py
│   │   │   ├── __init__.py
│   │   ├── __init__.py
│   ├── timescaledb/         # ماژول TimeScaleDB
│── tests/                   # تست‌ها
│── docker-compose.yml        # مدیریت سرویس‌ها در Docker
│── .env                      # متغیرهای محیطی
│── README.md                 # مستندات
```

---

## 3. مراحل پیاده‌سازی

### ✅ مرحله 1: ایجاد پوشه `Prometheus` در `infrastructure/`
```bash
mkdir -p infrastructure/prometheus/{adapters,config,monitoring,service}
```

### ✅ مرحله 2: انتقال فایل `prometheus.yml` به مسیر صحیح
```bash
mv prometheus.yml infrastructure/prometheus/config/
```

### ✅ مرحله 3: نوشتن `config/prometheus_config.py`
```python
import os

class PrometheusConfig:
    PORT = int(os.getenv("PROMETHEUS_PORT", 9090))
    METRICS_PATH = os.getenv("PROMETHEUS_METRICS_PATH", "/metrics")
```

### ✅ مرحله 4: نوشتن `monitoring/prometheus_exporter.py`
```python
from prometheus_client import CollectorRegistry, Gauge, Counter, start_http_server
import threading
from ..config.prometheus_config import PrometheusConfig

class PrometheusExporter:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.query_time_gauge = Gauge("clickhouse_query_time", "Execution time of ClickHouse queries", registry=self.registry)
        self.request_counter = Counter("http_requests_total", "Total number of HTTP requests", registry=self.registry)

    def start_exporter(self):
        thread = threading.Thread(target=start_http_server, args=(PrometheusConfig.PORT,))
        thread.daemon = True
        thread.start()
```

### ✅ مرحله 5: نوشتن `service/metrics_service.py`
```python
from ..monitoring.prometheus_exporter import PrometheusExporter

class MetricsService:
    def __init__(self, exporter: PrometheusExporter):
        self.exporter = exporter

    def record_query_time(self, duration):
        self.exporter.query_time_gauge.set(duration)

    def increment_request_count(self):
        self.exporter.request_counter.inc()
```

---

## 4. راه‌اندازی مجدد `Prometheus` با این ساختار

### 🔹 **حذف سرویس قدیمی `Prometheus`**
```bash
docker-compose down
docker rm smart_whale-prometheus-1
docker rmi prom/prometheus:v2.30.3
```

### 🔹 **اجرای `Prometheus` با ساختار جدید**
```bash
docker-compose up -d --force-recreate --build prometheus
```

### 🔹 **بررسی وضعیت `Prometheus`**
```bash
docker ps | grep prometheus
docker logs smart_whale-prometheus-1 --tail=50
```

---

## 5. بررسی نهایی و تست
🚀 **حالا `Prometheus` مانند `Kafka`، `Redis` و `ClickHouse` کاملاً ماژولار شده است!**
✅ **تست کن که همه چیز درست کار می‌کند و اگر جایی مشکل داشت، اطلاع بده تا اصلاح کنیم.** 😊

