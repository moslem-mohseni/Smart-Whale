import json
import logging
from pymilvus import Collection
from ..config.config import config as collection_config
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.security.encryption import FileEncryption

# تنظیمات لاگ‌ها
logging.basicConfig(filename="restore.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RestoreService:
    """مدیریت بازیابی داده‌های پشتیبان از MinIO"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or collection_config.DEFAULT_COLLECTION_NAME
        self.file_service = FileService()
        self.encryption_service = FileEncryption()

    def restore_from_backup(self, backup_file_id: str):
        """بازیابی داده‌ها از نسخه پشتیبان ذخیره‌شده در MinIO"""
        try:
            encrypted_data = self.file_service.download_file(backup_file_id)
            if not encrypted_data:
                logging.error(f"❌ فایل پشتیبان `{backup_file_id}` یافت نشد.")
                print(f"❌ فایل پشتیبان یافت نشد: {backup_file_id}")
                return

            # رمزگشایی داده‌های پشتیبان
            decrypted_data = self.encryption_service.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode())

            # درج داده‌ها در Collection
            collection = Collection(self.collection_name)
            collection.insert([[entry["id"] for entry in data], [entry["vector"] for entry in data], [entry["metadata"] for entry in data]])

            logging.info(f"✅ بازیابی داده‌ها از پشتیبان `{backup_file_id}` به Collection `{self.collection_name}` انجام شد.")
            print(f"✅ بازیابی داده‌ها از `{backup_file_id}` انجام شد.")
        except Exception as e:
            logging.error(f"❌ خطا در بازیابی پشتیبان: {e}")
            print(f"❌ خطا در بازیابی پشتیبان: {e}")
