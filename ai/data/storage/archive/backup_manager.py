import os
import shutil
from datetime import datetime
from infrastructure.file_management.adapters.minio_adapter import MinIOAdapter
from infrastructure.clickhouse.adapters.clickhouse_adapter import ClickHouseAdapter
from infrastructure.elasticsearch.adapters.elasticsearch_adapter import ElasticsearchAdapter
from typing import Optional

class BackupManager:
    """
    مدیریت پشتیبان‌گیری و بازیابی داده‌ها.
    """

    def __init__(self, backup_dir: str = "./backups", minio_bucket: str = "backups"):
        """
        مقداردهی اولیه مدیریت پشتیبان‌گیری.

        :param backup_dir: مسیر ذخیره‌سازی فایل‌های پشتیبان محلی
        :param minio_bucket: نام باکت MinIO برای ذخیره‌سازی پشتیبان‌ها
        """
        self.backup_dir = backup_dir
        self.minio_bucket = minio_bucket
        self.minio_adapter = MinIOAdapter()
        self.clickhouse_adapter = ClickHouseAdapter()
        self.elasticsearch_adapter = ElasticsearchAdapter()

        # اطمینان از ایجاد مسیر پشتیبان‌گیری
        os.makedirs(self.backup_dir, exist_ok=True)

    async def connect(self) -> None:
        """ اتصال به سرویس‌های مورد نیاز. """
        await self.minio_adapter.connect()
        await self.clickhouse_adapter.connect()
        await self.elasticsearch_adapter.connect()

    async def create_backup(self) -> Optional[str]:
        """
        ایجاد نسخه پشتیبان از پایگاه داده‌ها و ذخیره در MinIO.

        :return: مسیر فایل پشتیبان یا None در صورت خطا
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"backup_{timestamp}.zip")

        try:
            # پشتیبان‌گیری از ClickHouse
            clickhouse_backup = os.path.join(self.backup_dir, f"clickhouse_backup_{timestamp}.sql")
            await self.clickhouse_adapter.backup_database(clickhouse_backup)

            # پشتیبان‌گیری از Elasticsearch
            elastic_backup = os.path.join(self.backup_dir, f"elastic_backup_{timestamp}.json")
            await self.elasticsearch_adapter.backup_data(elastic_backup)

            # فشرده‌سازی نسخه‌های پشتیبان
            shutil.make_archive(backup_file.replace(".zip", ""), "zip", self.backup_dir)

            # آپلود فایل پشتیبان در MinIO
            with open(backup_file, "rb") as file_data:
                await self.minio_adapter.upload_file(self.minio_bucket, os.path.basename(backup_file), file_data.read())

            # حذف فایل‌های موقت
            os.remove(clickhouse_backup)
            os.remove(elastic_backup)

            return backup_file
        except Exception as e:
            print(f"⚠️ خطا در ایجاد پشتیبان: {e}")
            return None

    async def restore_backup(self, backup_filename: str) -> bool:
        """
        بازیابی داده‌ها از یک نسخه پشتیبان.

        :param backup_filename: نام فایل پشتیبان در MinIO
        :return: True در صورت موفقیت، False در صورت خطا
        """
        try:
            # دریافت فایل پشتیبان از MinIO
            backup_file = os.path.join(self.backup_dir, backup_filename)
            backup_data = await self.minio_adapter.download_file(self.minio_bucket, backup_filename)

            with open(backup_file, "wb") as file:
                file.write(backup_data)

            # استخراج فایل‌ها
            shutil.unpack_archive(backup_file, self.backup_dir)

            # بازیابی داده‌ها در ClickHouse و Elasticsearch
            clickhouse_backup = os.path.join(self.backup_dir, f"clickhouse_backup_{backup_filename.replace('.zip', '.sql')}")
            elastic_backup = os.path.join(self.backup_dir, f"elastic_backup_{backup_filename.replace('.zip', '.json')}")

            await self.clickhouse_adapter.restore_database(clickhouse_backup)
            await self.elasticsearch_adapter.restore_data(elastic_backup)

            # حذف فایل‌های موقت
            os.remove(clickhouse_backup)
            os.remove(elastic_backup)
            os.remove(backup_file)

            return True
        except Exception as e:
            print(f"⚠️ خطا در بازیابی پشتیبان: {e}")
            return False

    async def close(self) -> None:
        """ قطع اتصال از سرویس‌های مورد نیاز. """
        await self.minio_adapter.disconnect()
        await self.clickhouse_adapter.disconnect()
        await self.elasticsearch_adapter.disconnect()
