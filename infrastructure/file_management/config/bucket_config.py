import os

class BucketConfig:
    """
    مدیریت تنظیمات مربوط به باکت‌های MinIO
    """
    DEFAULT_BUCKET = os.getenv("MINIO_BUCKET_NAME", "default")
    ARCHIVE_BUCKET = os.getenv("MINIO_ARCHIVE_BUCKET", "archive")
    TEMP_BUCKET = os.getenv("MINIO_TEMP_BUCKET", "temp")