from minio import Minio
from minio.error import S3Error
import os


class MinioAdapter:
    def __init__(self):
        self.client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "your_access_key"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "your_secret_key"),
            secure=False
        )
        self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "default")
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """بررسی و ایجاد خودکار bucket در MinIO"""
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def upload_file(self, file_data, file_name, content_type):
        """آپلود فایل در MinIO"""
        try:
            self.client.put_object(
                self.bucket_name, file_name, file_data, len(file_data), content_type=content_type
            )
            return {"status": "success", "file_name": file_name}
        except S3Error as e:
            return {"status": "error", "message": str(e)}

    def download_file(self, file_name):
        """دریافت فایل از MinIO"""
        try:
            response = self.client.get_object(self.bucket_name, file_name)
            return response.read()
        except S3Error as e:
            return {"status": "error", "message": str(e)}

    def delete_file(self, file_name):
        """حذف فایل از MinIO"""
        try:
            self.client.remove_object(self.bucket_name, file_name)
            return {"status": "success", "file_name": file_name}
        except S3Error as e:
            return {"status": "error", "message": str(e)}

    def file_exists(self, file_name):
        """بررسی وجود فایل در MinIO"""
        try:
            self.client.stat_object(self.bucket_name, file_name)
            return True
        except S3Error:
            return False
