# مستندات ماژول Vector Store (Milvus)

## 🎯 هدف ماژول

📂 **مسیر ماژول:** `infrastructure/vector_store/`

ماژول `Vector Store` با استفاده از Milvus برای **مدیریت و جستجوی بردارها** طراحی شده است. قابلیت‌های اصلی:
- ذخیره‌سازی بردارهای با ابعاد بالا
- جستجوی Approximate Nearest Neighbor (ANN)
- مقیاس‌پذیری افقی و عمودی
- مدیریت متادیتا و فیلدهای اضافی
- پشتیبان‌گیری و بازیابی خودکار

## 📂 ساختار ماژول
```
vector_store/
├── __init__.py
│
├── adapters/
│   ├── milvus_adapter.py          # اتصال به Milvus
│   ├── retry_mechanism.py         # مکانیزم تلاش مجدد
│   └── connection_pool.py         # مدیریت Pool اتصالات
│
├── config/
│   ├── settings.py                # تنظیمات از .env
│   ├── collection_config.py       # تنظیمات Collection ها
│   └── index_config.py            # تنظیمات ایندکس‌ها
│
├── domain/
│   ├── models.py                  # مدل‌های داده
│   ├── vector.py                  # کلاس Vector
│   └── collection.py              # کلاس Collection
│
├── service/
│   ├── vector_service.py          # سرویس اصلی
│   ├── index_service.py           # مدیریت ایندکس‌ها
│   └── search_service.py          # جستجوی بردارها
│
├── optimization/
│   ├── cache_manager.py           # مدیریت کش
│   ├── batch_processor.py         # پردازش دسته‌ای
│   └── query_optimizer.py         # بهینه‌سازی کوئری
│
├── monitoring/
│   ├── metrics.py                 # متریک‌های عملکردی
│   ├── health_check.py           # بررسی سلامت
│   └── performance_logger.py      # ثبت کارایی
│
├── backup/
│   ├── backup_service.py          # پشتیبان‌گیری
│   ├── restore_service.py         # بازیابی
│   └── scheduler.py               # زمان‌بندی
│
└── migrations/
    ├── manager.py                 # مدیریت مهاجرت‌ها
    └── versions/                  # نسخه‌های مهاجرت
```

## ⚙️ کلاس‌های اصلی و متدها

### 1. VectorService
```python
class VectorService:
    async def insert_vectors(collection_name: str, vectors: List[Vector])
    async def search_vectors(collection_name: str, query_vector: Vector, top_k: int)
    async def batch_insert(collection_name: str, vectors: List[Vector])
    async def delete_vectors(collection_name: str, ids: List[str])
```

### 2. IndexService
```python
class IndexService:
    async def create_index(collection_name: str, index_type: str)
    async def drop_index(collection_name: str)
    async def rebuild_index(collection_name: str)
```

### 3. SearchService
```python
class SearchService:
    async def similarity_search(vector: Vector, top_k: int)
    async def range_search(vector: Vector, radius: float)
    async def hybrid_search(vector: Vector, filters: Dict)
```

## 🔧 تنظیمات محیطی (.env)
```env
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=root
MILVUS_PASSWORD=milvus
MILVUS_COLLECTION_SHARDS=2
MILVUS_REPLICA_FACTOR=1
```

## 🔄 مقیاس‌پذیری
- Sharding خودکار Collection ها
- Read Replicas برای توزیع بار
- Connection Pool برای مدیریت اتصالات
- پردازش دسته‌ای برای عملیات حجیم

## 🔒 امنیت
- احراز هویت با JWT
- RBAC برای کنترل دسترسی
- رمزنگاری داده‌ها در حال انتقال
- Audit Logging

## 📊 مانیتورینگ
- متریک‌های Prometheus
- Health Check endpoints
- Performance logging
- Query profiling

## 💾 پشتیبان‌گیری
- پشتیبان‌گیری دوره‌ای خودکار
- بازیابی نقطه‌ای
- مهاجرت داده‌ها بین نسخه‌ها
- حفظ سازگاری در هنگام بازیابی

## 🔍 بهینه‌سازی
- کش نتایج جستجو در Redis
- پردازش دسته‌ای برای عملیات bulk
- ایندکس‌گذاری هوشمند
- Query optimization