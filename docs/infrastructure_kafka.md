# مستندات ماژول Kafka

## هدف ماژول

📂 **مسیر ماژول:** `infrastructure/kafka/`

ماژول `Kafka` برای مدیریت ارتباط با سرور Kafka در پروژه طراحی شده است. این ماژول شامل تولیدکننده و مصرف‌کننده پیام، مدیریت موضوعات، مهاجرت‌های مربوط به Kafka، و سرویس‌های نگهداری و پایش است.

---

## **ساختار ماژول**

```
kafka/
    │── __init__.py                # مقداردهی اولیه ماژول
    │
    ├── adapters/
    │   ├── __init__.py            # مقداردهی اولیه آداپتورها
    │   ├── backpressure.py       # مدیریت فشار بار مصرف‌کننده‌ها
    │   ├── circuit_breaker.py    # مدیریت خطاها با مکانیزم Circuit Breaker
    │   ├── connection_pool.py    # مدیریت اتصال به Kafka
    │   ├── consumer.py          # کلاس مصرف‌کننده پیام Kafka
    │   ├── producer.py          # کلاس تولیدکننده پیام Kafka
    │   └── retry_mechanism.py   # مدیریت ارسال مجدد پیام‌ها در صورت خطا
    │
    ├── config/
    │   ├── __init__.py            # مقداردهی اولیه تنظیمات
    │   └── settings.py            # تنظیمات اتصال به Kafka
    │
    ├── domain/
    │   ├── __init__.py            # مقداردهی اولیه مدل‌ها
    │   └── models.py              # مدل‌های داده‌ای مربوط به Kafka
    │
    ├── migrations/
    │   ├── __init__.py            # مقداردهی اولیه مهاجرت‌ها
    │   └── v001_create_base_topics.py # مدیریت موضوعات اولیه Kafka
    │
    ├── monitoring/
    │   ├── __init__.py            # مقداردهی اولیه پایش
    │   ├── alerts.py              # مدیریت هشدارهای Kafka
    │   ├── health_check.py        # بررسی سلامت Kafka
    │   └── metrics.py             # ثبت متریک‌های عملکرد Kafka
    │
    ├── scripts/
    │   ├── __init__.py            # مقداردهی اولیه اسکریپت‌ها
    │   ├── maintenance.py         # عملیات نگهداری Kafka
    │   └── topic_manager.py       # مدیریت موضوعات Kafka
    │
    └── service/
        ├── __init__.py            # مقداردهی اولیه سرویس‌ها
        ├── batch_processor.py     # پردازش دسته‌ای پیام‌ها
        ├── kafka_service.py       # سرویس مرکزی مدیریت Kafka
        ├── message_cache.py       # کش کردن پیام‌ها
        └── partition_manager.py   # مدیریت پارتیشن‌ها
```

---

## **شرح فایل‌ها و عملکرد آن‌ها**

### **1️⃣ `consumer.py` - مصرف‌کننده پیام Kafka**
📌 **هدف:** دریافت و پردازش پیام‌های ورودی از Kafka.

**کلاس‌ها و متدها:**

#### `MessageConsumer`
- `__init__(config: KafkaConfig)`: مقداردهی اولیه مصرف‌کننده
- `async consume(topic: str, group_id: str, handler: Callable[[Message], Awaitable[None]]) -> None`: اشتراک در موضوع و پردازش پیام‌ها
- `async stop() -> None`: توقف مصرف‌کننده

---

### **2️⃣ `producer.py` - تولیدکننده پیام Kafka**
📌 **هدف:** ارسال پیام‌ها به Kafka.

**کلاس‌ها و متدها:**

#### `MessageProducer`
- `__init__(config: KafkaConfig)`: مقداردهی اولیه تولیدکننده
- `async send(message: Message) -> None`: ارسال پیام به Kafka
- `async send_batch(messages: List[Message]) -> None`: ارسال دسته‌ای پیام‌ها

---

### **3️⃣ `settings.py` - تنظیمات Kafka**
📌 **هدف:** مدیریت پارامترهای اتصال به Kafka.

**کلاس‌ها و متدها:**

#### `KafkaConfig`
- `get_producer_config() -> dict`: تنظیمات تولیدکننده
- `get_consumer_config() -> dict`: تنظیمات مصرف‌کننده
- `get_security_config() -> dict`: تنظیمات امنیتی

---

### **4️⃣ `models.py` - مدل‌های داده‌ای**
📌 **هدف:** تعریف مدل‌های مرتبط با پیام‌ها و موضوعات Kafka.

**کلاس‌ها:**

- `Message`: مدل پیام‌های Kafka شامل `topic`، `content`، `metadata`
- `TopicConfig`: تنظیمات موضوع شامل تعداد پارتیشن‌ها و فاکتور تکرار

---

### **5️⃣ `kafka_service.py` - سرویس مرکزی مدیریت Kafka**
📌 **هدف:** نقطه مرکزی مدیریت تولید و مصرف پیام‌ها.

**کلاس:**

#### `KafkaService`
- `async send_message(message: Message) -> None`: ارسال پیام به Kafka
- `async send_messages_batch(messages: List[Message]) -> None`: ارسال دسته‌ای پیام‌ها
- `async subscribe(topic: str, group_id: str, handler: Callable[[Message], Awaitable[None]]) -> None`: اشتراک در یک موضوع
- `async stop_all() -> None`: توقف تمام مصرف‌کننده‌ها
- `async create_topic(topic_config: TopicConfig) -> None`: ایجاد یک موضوع جدید در Kafka

---

### **6️⃣ `monitoring/` - پایش Kafka**
📌 **هدف:** بررسی عملکرد Kafka و سلامت سرویس‌ها.

**فایل‌ها:**
- `alerts.py`: هشدارهای مربوط به Kafka
- `health_check.py`: بررسی وضعیت سلامت Kafka
- `metrics.py`: ثبت متریک‌های Kafka

---

## **نحوه استفاده از ماژول در پروژه**

```python
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.kafka.config.settings import KafkaConfig
from infrastructure.kafka.domain.models import Message

config = KafkaConfig()
kafka_service = KafkaService(config)

message = Message(topic="test_topic", content="Hello Kafka!")
await kafka_service.send_message(message)
```

