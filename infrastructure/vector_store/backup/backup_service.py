import json
import logging
from datetime import datetime
from pymilvus import Collection
from ..config.config import config as collection_config
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.security.encryption import FileEncryption
from infrastructure.file_management.domain.hash_service import HashService

# تنظیمات لاگ‌ها
logging.basicConfig(filename="backup.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BackupService:
    """مدیریت پشتیبان‌گیری از Collectionهای Milvus با استفاده از ماژول مدیریت فایل"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or collection_config.DEFAULT_COLLECTION_NAME
        self.file_service = FileService()
        self.encryption_service = FileEncryption()
        self.hash_service = HashService()

    def create_backup(self):
        """ایجاد نسخه پشتیبان از Collection و ذخیره در MinIO"""
        try:
            collection = Collection(self.collection_name)
            entities = collection.query(expr="", output_fields=["id", "vector", "metadata"])
            backup_data = json.dumps(entities, indent=4).encode()

            # محاسبه هش برای بررسی داده‌های تکراری
            backup_hash = self.hash_service.calculate_hash(backup_data)

            # بررسی اینکه آیا این نسخه قبلاً ذخیره شده است یا خیر
            stored_files = self.file_service.list_files()
            for file in stored_files:
                if file["name"] == backup_hash:
                    logging.info(f"⚠️ نسخه پشتیبان '{self.collection_name}' قبلاً ذخیره شده است.")
                    print(f"⚠️ نسخه پشتیبان جدیدی ایجاد نشد، زیرا همین نسخه قبلاً ذخیره شده است.")
                    return file["file_id"]

            # رمزنگاری داده‌های پشتیبان
            encrypted_data = self.encryption_service.encrypt(backup_data)

            # ذخیره‌سازی در MinIO
            backup_filename = f"{backup_hash}.enc"
            file_id = self.file_service.upload_file(encrypted_data, backup_filename)

            logging.info(f"✅ نسخه پشتیبان با شناسه `{file_id}` ذخیره شد.")
            print(f"✅ نسخه پشتیبان با موفقیت ذخیره شد. شناسه فایل: {file_id}")
            return file_id
        except Exception as e:
            logging.error(f"❌ خطا در ایجاد پشتیبان: {e}")
            print(f"❌ خطا در ایجاد پشتیبان: {e}")
            return None
