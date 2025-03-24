import logging
from typing import Optional
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache
from infrastructure.file_management.storage.lifecycle import LifecycleManager

class FileStore:
    """
    این کلاس مدیریت ذخیره‌سازی فایل‌های پردازشی زبان را بر عهده دارد.
    """

    def __init__(self, file_service: FileService, hash_cache: HashCache, lifecycle_manager: LifecycleManager):
        self.file_service = file_service
        self.hash_cache = hash_cache
        self.lifecycle_manager = lifecycle_manager
        logging.info("✅ FileStore مقداردهی شد و ارتباط با سرویس‌های زیرساختی برقرار شد.")

    async def save_file(self, file_name: str, file_data: bytes) -> Optional[str]:
        """
        ذخیره‌سازی فایل با بررسی تکراری نبودن و مدیریت چرخه‌ی حیات.

        :param file_name: نام فایل
        :param file_data: محتوای فایل
        :return: شناسه‌ی فایل در صورت موفقیت
        """
        try:
            # بررسی هش فایل برای جلوگیری از ذخیره‌سازی فایل‌های تکراری
            file_hash = self.hash_cache.calculate_hash(file_data)
            existing_file = self.hash_cache.get_file_hash(file_name)

            if existing_file and existing_file == file_hash:
                logging.warning(f"⚠️ فایل `{file_name}` قبلاً در سیستم ذخیره شده است.")
                return None

            # ذخیره‌سازی فایل
            file_id = await self.file_service.upload_file(file_name, file_data)

            # ذخیره هش فایل در کش برای Deduplication
            self.hash_cache.store_file_hash(file_name, file_hash)

            logging.info(f"✅ فایل `{file_name}` ذخیره شد. [File ID: {file_id}]")
            return file_id

        except Exception as e:
            logging.error(f"❌ خطا در ذخیره‌سازی فایل `{file_name}`: {e}")
            return None

    async def retrieve_file(self, file_id: str) -> Optional[bytes]:
        """
        بازیابی فایل از سیستم ذخیره‌سازی.

        :param file_id: شناسه‌ی فایل در سیستم
        :return: محتوای فایل به‌صورت `bytes`
        """
        try:
            file_data = await self.file_service.download_file(file_id)

            if not file_data:
                logging.warning(f"⚠️ فایل با شناسه `{file_id}` در سیستم یافت نشد.")
                return None

            logging.info(f"📥 فایل `{file_id}` دریافت شد.")
            return file_data

        except Exception as e:
            logging.error(f"❌ خطا در دریافت فایل `{file_id}`: {e}")
            return None

    async def remove_file(self, file_id: str):
        """
        حذف فایل از سیستم ذخیره‌سازی.

        :param file_id: شناسه‌ی فایل در سیستم
        """
        try:
            await self.file_service.delete_file(file_id)
            logging.info(f"🗑 فایل `{file_id}` با موفقیت از سیستم حذف شد.")

        except Exception as e:
            logging.error(f"❌ خطا در حذف فایل `{file_id}`: {e}")

    async def cleanup_old_files(self):
        """
        اجرای سیاست‌های چرخه‌ی حیات و حذف فایل‌های قدیمی.
        """
        try:
            await self.lifecycle_manager.cleanup_old_files()
            logging.info("🗑 فایل‌های قدیمی حذف شدند.")

        except Exception as e:
            logging.error(f"❌ خطا در حذف فایل‌های قدیمی: {e}")
