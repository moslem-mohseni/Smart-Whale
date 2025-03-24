"""
ماژول `file_management/` وظیفه‌ی مدیریت ذخیره‌سازی، دریافت، و حذف فایل‌های پردازشی زبان را بر عهده دارد.

📌 اجزای اصلی این ماژول:
- `file_service.py` → سرویس آپلود، دانلود و حذف فایل‌ها
- `file_store.py` → مدیریت ذخیره‌سازی و چرخه‌ی حیات فایل‌ها
"""

from .file_service import FileManagementService
from .file_store import FileStore
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache
from infrastructure.file_management.storage.lifecycle import LifecycleManager
from infrastructure.file_management.security.encryption import EncryptionService

# مقداردهی اولیه سرویس‌های مورد نیاز
file_service = FileService()
hash_cache = HashCache()
lifecycle_manager = LifecycleManager()
encryption_service = EncryptionService()

# مقداردهی اولیه FileManagementService و FileStore
file_management_service = FileManagementService(file_service, hash_cache, encryption_service)
file_store = FileStore(file_service, hash_cache, lifecycle_manager)

__all__ = [
    "file_management_service",
    "file_store",
    "FileManagementService",
    "FileStore",
]
