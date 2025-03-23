# مستندات ماژول File Management

## 🎯 هدف ماژول

📂 **مسیر ماژول:** `infrastructure/file_management/`

ماژول `File Management` برای **مدیریت، ذخیره‌سازی و پردازش فایل‌ها** در سیستم طراحی شده است. این ماژول از **MinIO** به‌عنوان سیستم ذخیره‌سازی ابری استفاده می‌کند و قابلیت‌های زیر را ارائه می‌دهد:

- **مدیریت فایل‌های حجیم** با پشتیبانی از **Multipart Upload**
- **تشخیص و جلوگیری از ذخیره‌سازی فایل‌های تکراری (Deduplication)**
- **مدیریت دسترسی کاربران** با استفاده از **RBAC و JWT**
- **رمزنگاری، فشرده‌سازی و ایمن‌سازی فایل‌ها**
- **کشینگ و بهینه‌سازی برای افزایش سرعت دسترسی**
- **یکپارچه‌سازی با سیستم‌های Kafka و Redis برای پردازش‌های ناهمگام**
- **سیستم مانیتورینگ و ثبت متریک‌های عملکردی**

---

## 📂 **ساختار دایرکتوری ماژول**
```
file_management/
│── __init__.py                    # مقداردهی اولیه ماژول
│
├── adapters/                       # مدیریت ارتباط با MinIO و زیرساخت‌ها
│   ├── minio_adapter.py            # آداپتور ارتباط با MinIO
│   ├── retry_mechanism.py          # مدیریت Retry برای عملیات ناموفق
│   ├── circuit_breaker.py          # مدیریت Circuit Breaker برای جلوگیری از فشار بر سیستم
│   ├── connection_pool.py          # مدیریت اتصال‌های MinIO
│
├── cache/                          # سیستم کش برای افزایش سرعت پردازش
│   ├── cache_manager.py            # مدیریت کش متادیتای فایل‌ها
│   ├── hash_cache.py               # ذخیره هش‌های فایل‌ها برای Deduplication
│
├── config/                         # تنظیمات و پیکربندی
│   ├── settings.py                 # تنظیمات کلی سیستم
│   ├── bucket_config.py            # تنظیمات مربوط به باکت‌های MinIO
│
├── domain/                         # مدل‌های داده‌ای و اشیای مقدار
│   ├── models.py                   # مدل‌های مرتبط با فایل
│   ├── value_objects.py            # اشیای مقدار و انواع داده‌ای
│   ├── file_metadata.py            # مدیریت متادیتای فایل‌ها
│   ├── hash_service.py             # مدیریت و محاسبه هش فایل‌ها
│
├── monitoring/                     # سیستم مانیتورینگ و متریک‌های عملکردی
│   ├── metrics.py                  # جمع‌آوری متریک‌های مربوط به مدیریت فایل
│   ├── health_check.py             # بررسی سلامت سرویس و اتصال به MinIO
│
├── security/                        # مدیریت امنیت و کنترل دسترسی
│   ├── access_control.py           # کنترل دسترسی به فایل‌ها بر اساس JWT
│   ├── file_validator.py           # بررسی نوع، اندازه و امنیت فایل‌ها
│   ├── sanitizer.py                # پاکسازی نام و محتوای فایل از داده‌های مخرب
│   ├── encryption.py               # رمزنگاری فایل‌ها
│
├── service/                         # پیاده‌سازی سرویس‌های اصلی مدیریت فایل
│   ├── file_service.py             # سرویس آپلود، دانلود و حذف فایل‌ها
│   ├── search_service.py           # سرویس جستجوی فایل‌ها
│   ├── compression.py              # فشرده‌سازی فایل‌ها
│
├── storage/                         # مدیریت ذخیره‌سازی و چرخه حیات فایل‌ها
│   ├── file_store.py               # سیستم ذخیره‌سازی فایل‌ها
│   ├── deduplication.py            # مدیریت و تشخیص فایل‌های تکراری
│   ├── lifecycle.py                # سیستم مدیریت چرخه حیات فایل‌ها
```

---

## ✅ **ویژگی‌های کلیدی ماژول**
✔ **ذخیره‌سازی و بازیابی سریع فایل‌ها** – استفاده از **MinIO** برای مقیاس‌پذیری بهتر  
✔ **مدیریت فایل‌های تکراری** – تشخیص فایل‌های مشابه بر اساس **هش‌گذاری**  
✔ **امنیت و کنترل دسترسی** – احراز هویت **JWT** و **رمزنگاری فایل‌ها**  
✔ **بهینه‌سازی عملکرد** – کشینگ با **Redis**، فشرده‌سازی و مدیریت چرخه حیات  
✔ **مدیریت دسترسی سطح بالا** – **RBAC** برای کاربران و **Audit Logging** برای ثبت عملیات  

---



# 📄 مستندات فنی ماژول File Management

## 📌 معرفی بخش‌های ماژول
این بخش شامل توضیحات دقیق درباره **هر ماژول، کلاس و متد** در سیستم مدیریت فایل‌ها است.

---

## 🛠 **1. بخش `adapters/` (اتصال به MinIO و زیرساخت)**
### 🔹 `minio_adapter.py`
- **MinIOAdapter** → مدیریت ارتباط با MinIO
  - `upload_file(file_name, data)` → آپلود فایل به MinIO
  - `download_file(file_name)` → دریافت فایل از MinIO
  - `delete_file(file_name)` → حذف فایل از MinIO

### 🔹 `retry_mechanism.py`
- **RetryHandler** → اجرای مجدد عملیات در صورت بروز خطا
  - `execute_with_retry(operation, retries=3)` → اجرای تابع با قابلیت Retry

### 🔹 `circuit_breaker.py`
- **CircuitBreaker** → جلوگیری از ارسال درخواست‌های مکرر در صورت بروز خطا
  - `call(operation)` → اجرای عملیات با Circuit Breaker

---

## ⚡ **2. بخش `cache/` (مدیریت کش و افزایش سرعت)**
### 🔹 `cache_manager.py`
- **CacheManager** → مدیریت کش با Redis
  - `set(key, value, ttl=3600)` → ذخیره مقدار در کش
  - `get(key)` → دریافت مقدار از کش
  - `delete(key)` → حذف مقدار از کش

### 🔹 `hash_cache.py`
- **HashCache** → مدیریت هش فایل‌ها در کش برای جلوگیری از ذخیره فایل‌های تکراری
  - `store_file_hash(file_name, file_hash)` → ذخیره هش فایل در کش
  - `get_file_hash(file_name)` → دریافت هش فایل از کش

---

## ⚙️ **3. بخش `config/` (تنظیمات و پیکربندی)**
### 🔹 `settings.py`
- مدیریت تنظیمات **MinIO، Kafka، Redis و محدودیت‌های فایل‌ها**

### 🔹 `bucket_config.py`
- مدیریت و پیکربندی **باکت‌های MinIO**

---

## 📂 **4. بخش `domain/` (مدل‌های داده‌ای و اشیای مقدار)**
### 🔹 `models.py`
- **FileModel** → مدل داده‌ای اصلی فایل
  - ویژگی‌ها: `id`، `name`، `size`، `type`، `created_at`

### 🔹 `file_metadata.py`
- **FileMetadata** → متادیتای فایل‌ها شامل نوع، اندازه و زمان ایجاد

### 🔹 `hash_service.py`
- **HashService** → محاسبه هش فایل برای Deduplication
  - `calculate_hash(file_data)` → محاسبه `SHA-256` از داده‌های فایل

---

## 🔍 **5. بخش `service/` (مدیریت فایل‌ها و پردازش)**
### 🔹 `file_service.py`
- **FileService** → مدیریت فایل‌ها (آپلود، دانلود، حذف)
  - `upload_file(file_data)` → ذخیره فایل با بررسی هش و کنترل دسترسی
  - `download_file(file_id)` → دریافت فایل با بررسی مجوز دسترسی
  - `delete_file(file_id)` → حذف فایل با ثبت گزارش لاگ

### 🔹 `search_service.py`
- **SearchService** → جستجوی فایل‌ها
  - `find_by_name(file_name)` → جستجوی فایل با نام
  - `find_by_hash(file_hash)` → جستجوی فایل بر اساس هش

### 🔹 `compression.py`
- **CompressionService** → فشرده‌سازی و استخراج فایل‌ها
  - `compress_file(file_data)` → فشرده‌سازی داده‌های فایل
  - `extract_file(compressed_data)` → استخراج فایل فشرده‌شده

---

## 🔒 **6. بخش `security/` (امنیت، کنترل دسترسی و رمزنگاری)**
### 🔹 `access_control.py`
- **AccessControl** → مدیریت نقش‌های کاربران با JWT

### 🔹 `file_validator.py`
- **FileValidator** → بررسی نوع و اندازه فایل قبل از ذخیره

### 🔹 `encryption.py`
- **EncryptionService** → رمزنگاری و رمزگشایی فایل‌ها

---

## 🏗 **7. بخش `storage/` (ذخیره‌سازی و مدیریت داده‌ها)**
### 🔹 `file_store.py`
- **FileStore** → مدیریت ساختار دایرکتوری و ذخیره‌سازی فایل‌ها

### 🔹 `deduplication.py`
- **DeduplicationService** → بررسی و حذف فایل‌های تکراری

### 🔹 `lifecycle.py`
- **LifecycleManager** → مدیریت چرخه حیات و حذف فایل‌های قدیمی

---



# 📄 نمونه استفاده از ماژول File Management

## 📌 راهنمای استفاده از سرویس‌ها و متدهای اصلی

### 🚀 **۱. راه‌اندازی ماژول و تنظیمات اولیه**
ابتدا باید تنظیمات محیطی را بارگذاری کنیم:

```python
from infrastructure.file_management.config.settings import FileManagementSettings

# بارگذاری تنظیمات
settings = FileManagementSettings()
print(settings.MINIO_ENDPOINT)
```

---

### 📂 **۲. مدیریت فایل‌ها با `FileService`**

#### 📌 **آپلود فایل**
```python
from infrastructure.file_management.service.file_service import FileService

file_service = FileService()

with open("sample.txt", "rb") as file:
    file_data = file.read()

file_id = file_service.upload_file(file_data)
print(f"File uploaded successfully with ID: {file_id}")
```

#### 📌 **دانلود فایل**
```python
downloaded_file = file_service.download_file(file_id)
with open("downloaded_sample.txt", "wb") as file:
    file.write(downloaded_file)

print("File downloaded successfully!")
```

#### 📌 **حذف فایل**
```python
file_service.delete_file(file_id)
print("File deleted successfully!")
```

---

### 🔍 **۳. جستجوی فایل‌ها با `SearchService`**

#### 📌 **جستجو بر اساس نام فایل**
```python
from infrastructure.file_management.service.search_service import SearchService

search_service = SearchService()
results = search_service.find_by_name("sample.txt")

print("Search results:", results)
```

#### 📌 **جستجو بر اساس هش فایل**
```python
hash_results = search_service.find_by_hash("e3b0c44298fc1c149afbf4c8996fb924")
print("Hash search results:", hash_results)
```

---

### 🔐 **۴. رمزنگاری و امنیت فایل‌ها با `EncryptionService`**

#### 📌 **رمزنگاری فایل**
```python
from infrastructure.file_management.security.encryption import EncryptionService

encryption_service = EncryptionService()
encrypted_data = encryption_service.encrypt(file_data)

print("Encrypted Data:", encrypted_data)
```

#### 📌 **رمزگشایی فایل**
```python
decrypted_data = encryption_service.decrypt(encrypted_data)

print("Decrypted Data:", decrypted_data.decode("utf-8"))
```

---

### ⚡ **۵. مدیریت کش و جلوگیری از آپلود فایل‌های تکراری**

#### 📌 **بررسی هش فایل و جلوگیری از ذخیره‌سازی تکراری**
```python
from infrastructure.file_management.cache.hash_cache import HashCache

hash_cache = HashCache()
file_hash = "e3b0c44298fc1c149afbf4c8996fb924"  # نمونه هش

if hash_cache.get_file_hash("sample.txt"):
    print("File already exists in cache!")
else:
    hash_cache.store_file_hash("sample.txt", file_hash)
    print("File hash stored successfully!")
```

---

### 🎯 **۶. مدیریت چرخه حیات و حذف فایل‌های قدیمی**

```python
from infrastructure.file_management.storage.lifecycle import LifecycleManager

lifecycle_manager = LifecycleManager()
lifecycle_manager.cleanup_old_files()

print("Old files removed successfully!")
```

---

