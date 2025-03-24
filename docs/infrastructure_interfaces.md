# مستندات ماژول Interfaces

## هدف ماژول

📂 **مسیر ماژول:** `infrastructure/interfaces/`

ماژول `interfaces` مجموعه‌ای از اینترفیس‌ها و کلاس‌های انتزاعی را فراهم می‌کند که به عنوان پایه‌ای برای پیاده‌سازی ارتباطات مختلف مانند ذخیره‌سازی داده، کش، پیام‌رسانی و مدیریت خطاها در زیرساخت سیستم به کار می‌روند.

---

## **ساختار ماژول**

```
interfaces/
    │── __init__.py      # مقداردهی اولیه ماژول
    │
    ├── caching.py       # اینترفیس کش
    ├── exceptions.py    # کلاس‌های مدیریت خطا
    ├── messaging.py     # اینترفیس پیام‌رسانی
    └── storage.py       # اینترفیس ذخیره‌سازی داده
```

---

## **شرح فایل‌ها و عملکرد آن‌ها**

### **1️⃣ `__init__.py` - مقداردهی اولیه ماژول**
📌 **هدف:** تجمیع و تعریف `__all__` برای استفاده آسان‌تر از اینترفیس‌ها در بخش‌های مختلف پروژه.

**تعاریف اصلی:**
- `StorageInterface`
- `MessagingInterface`
- `CachingInterface`
- کلاس‌های مدیریت خطا (`InfrastructureError` و زیرکلاس‌های آن)

---

### **2️⃣ `caching.py` - اینترفیس کش**
📌 **هدف:** تعریف استانداردی برای سیستم‌های کش‌گذاری.

**کلاس:**
#### `CachingInterface`
- `async connect() -> None`: برقراری اتصال به سرور کش
- `async disconnect() -> None`: قطع اتصال از سرور کش
- `async get(key: str) -> Optional[Any]`: بازیابی مقدار از کش
- `async set(key: str, value: Any, ttl: Optional[int] = None) -> None`: ذخیره مقدار در کش
- `async delete(key: str) -> bool`: حذف مقدار از کش
- `async exists(key: str) -> bool`: بررسی وجود مقدار در کش
- `async ttl(key: str) -> Optional[int]`: دریافت زمان انقضای مقدار
- `async scan_keys(pattern: str) -> list`: جستجوی کلیدها با الگو

---

### **3️⃣ `exceptions.py` - کلاس‌های مدیریت خطا**
📌 **هدف:** مدیریت خطاهای زیرساختی و ارائه کلاس‌های عمومی برای کنترل بهتر مشکلات.

**کلاس‌های اصلی:**
- `InfrastructureError`: کلاس پایه برای تمام خطاهای زیرساختی
- `ConnectionError`: خطای مربوط به اتصال
- `OperationError`: خطای مربوط به عملیات ذخیره‌سازی یا پردازش
- `ConfigurationError`: خطای مربوط به تنظیمات نادرست
- `TimeoutError`: خطای مربوط به پایان مهلت زمانی عملیات
- `ValidationError`: خطای مربوط به اعتبارسنجی داده‌ها

---

### **4️⃣ `messaging.py` - اینترفیس پیام‌رسانی**
📌 **هدف:** تعریف استانداردی برای ارتباطات پیام‌رسانی بین سرویس‌ها (مثلاً Kafka).

**کلاس:**
#### `MessagingInterface`
- `async connect() -> None`: برقراری اتصال به سرور پیام‌رسان
- `async disconnect() -> None`: قطع اتصال از سرور پیام‌رسان
- `async publish(topic: str, message: Any) -> None`: انتشار پیام در یک موضوع مشخص
- `async subscribe(topic: str, handler: Callable[[Any], Awaitable[None]]) -> None`: اشتراک در یک موضوع برای دریافت پیام‌ها
- `async unsubscribe(topic: str) -> None`: لغو اشتراک از یک موضوع

---

### **5️⃣ `storage.py` - اینترفیس ذخیره‌سازی داده**
📌 **هدف:** تعریف عملیات استاندارد برای ذخیره‌سازی و مدیریت داده‌ها در پایگاه داده.

**کلاس:**
#### `StorageInterface`
- `async connect() -> None`: برقراری اتصال به پایگاه داده
- `async disconnect() -> None`: قطع اتصال از پایگاه داده
- `async is_connected() -> bool`: بررسی وضعیت اتصال
- `async execute(query: str, params: Optional[List[Any]] = None) -> List[Any]`: اجرای یک پرس‌وجو
- `async execute_many(query: str, params_list: List[List[Any]]) -> None`: اجرای چندین پرس‌وجو به صورت دسته‌ای
- `async begin_transaction() -> None`: شروع تراکنش
- `async commit() -> None`: تایید تراکنش
- `async rollback() -> None`: بازگردانی تراکنش
- `async create_table(table_name: str, schema: dict) -> None`: ایجاد جدول جدید
- `async create_hypertable(table_name: str, time_column: str) -> None`: تبدیل جدول به hypertable

---

## **نحوه استفاده از ماژول در پروژه**

### **۱. استفاده از اینترفیس کش**
```python
from infrastructure.interfaces.caching import CachingInterface

class RedisCache(CachingInterface):
    async def connect(self):
        print("اتصال به Redis برقرار شد")
```

### **۲. استفاده از اینترفیس پیام‌رسانی**
```python
from infrastructure.interfaces.messaging import MessagingInterface

class KafkaMessaging(MessagingInterface):
    async def connect(self):
        print("اتصال به Kafka برقرار شد")
```

### **۳. استفاده از اینترفیس ذخیره‌سازی**
```python
from infrastructure.interfaces.storage import StorageInterface

class PostgresStorage(StorageInterface):
    async def connect(self):
        print("اتصال به پایگاه داده برقرار شد")
```

---

این مستندات راهنمایی برای پیاده‌سازی ماژول `interfaces` فراهم می‌کند که پایه‌ای برای توسعه سایر بخش‌های زیرساخت پروژه خواهد بود.

