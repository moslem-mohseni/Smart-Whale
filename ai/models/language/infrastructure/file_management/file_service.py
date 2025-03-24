import logging
from typing import Optional
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache
from infrastructure.file_management.security.encryption import EncryptionService

class FileManagementService:
    """
    این کلاس مدیریت فایل‌های پردازشی زبان شامل ذخیره، دریافت و حذف فایل‌ها را بر عهده دارد.
    این ماژول به‌صورت مستقیم از `FileService` که در `infrastructure/file_management/` پیاده‌سازی شده است، استفاده می‌کند.
    """

    def __init__(self, file_service: FileService, hash_cache: HashCache, encryption_service: EncryptionService):
        self.file_service = file_service
        self.hash_cache = hash_cache
        self.encryption_service = encryption_service
        logging.info("✅ FileManagementService مقداردهی شد و ارتباط با سرویس‌های زیرساختی برقرار شد.")

    async def upload_file(self, file_name: str, file_data: bytes) -> Optional[str]:
        """
        آپلود فایل به سیستم ذخیره‌سازی.

        :param file_name: نام فایل
        :param file_data: محتوای فایل
        :return: شناسه‌ی فایل در صورت موفقیت
        """
        try:
            # بررسی هش فایل برای جلوگیری از ذخیره‌سازی فایل‌های تکراری
            file_hash = self.hash_cache.calculate_hash(file_data)
            existing_file = self.hash_cache.get_file_hash(file_name)

            if existing_file and existing_file == file_hash:
                logging.warning(f"⚠️ فایل `{file_name}` قبلاً در سیستم ذخیره شده است. از آپلود مجدد جلوگیری شد.")
                return None

            # رمزنگاری فایل قبل از ذخیره‌سازی
            encrypted_data = self.encryption_service.encrypt(file_data)

            # آپلود فایل
            file_id = await self.file_service.upload_file(file_name, encrypted_data)

            # ذخیره هش فایل در کش برای Deduplication
            self.hash_cache.store_file_hash(file_name, file_hash)

            logging.info(f"✅ فایل `{file_name}` با موفقیت در سیستم ذخیره شد. [File ID: {file_id}]")
            return file_id

        except Exception as e:
            logging.error(f"❌ خطا در آپلود فایل `{file_name}`: {e}")
            return None

    async def download_file(self, file_id: str) -> Optional[bytes]:
        """
        دانلود فایل از سیستم ذخیره‌سازی.

        :param file_id: شناسه‌ی فایل در سیستم
        :return: محتوای فایل به‌صورت `bytes`
        """
        try:
            encrypted_data = await self.file_service.download_file(file_id)

            if not encrypted_data:
                logging.warning(f"⚠️ فایل با شناسه `{file_id}` در سیستم یافت نشد.")
                return None

            # رمزگشایی فایل
            file_data = self.encryption_service.decrypt(encrypted_data)

            logging.info(f"📥 فایل `{file_id}` از سیستم دریافت شد.")
            return file_data

        except Exception as e:
            logging.error(f"❌ خطا در دریافت فایل `{file_id}`: {e}")
            return None

    async def delete_file(self, file_id: str):
        """
        حذف فایل از سیستم ذخیره‌سازی.

        :param file_id: شناسه‌ی فایل در سیستم
        """
        try:
            await self.file_service.delete_file(file_id)
            logging.info(f"🗑 فایل `{file_id}` با موفقیت از سیستم حذف شد.")

        except Exception as e:
            logging.error(f"❌ خطا در حذف فایل `{file_id}`: {e}")
