import os
import re


def parse_size(size_str):
    """تبدیل رشته‌های اندازه مانند '100MB' به عدد"""
    if isinstance(size_str, (int, float)):
        return int(size_str)

    if not isinstance(size_str, str):
        return 104857600  # 100MB پیش‌فرض

    pattern = r"^(\d+(\.\d+)?)\s*([KMG]B?)?$"
    match = re.match(pattern, size_str.strip(), re.IGNORECASE)

    if not match:
        return 104857600  # 100MB پیش‌فرض

    value = float(match.group(1))
    unit = match.group(3)

    if unit is None:
        return int(value)
    elif unit.upper() in ('K', 'KB'):
        return int(value * 1024)
    elif unit.upper() in ('M', 'MB'):
        return int(value * 1024 * 1024)
    elif unit.upper() in ('G', 'GB'):
        return int(value * 1024 * 1024 * 1024)
    else:
        return 104857600  # 100MB پیش‌فرض


class FileManagementSettings:
    """
    مدیریت تنظیمات مربوط به ماژول مدیریت فایل
    """
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "your_access_key")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "your_secret_key")
    MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "default")

    MAX_FILE_SIZE = parse_size(os.getenv("MAX_FILE_SIZE", "100MB"))
    ALLOWED_FILE_TYPES = os.getenv("ALLOWED_FILE_TYPES", "jpg,png,pdf,txt").split(",")

    FILE_STORAGE_PATH = os.getenv("FILE_STORAGE_PATH", "./storage")
    FILE_RETENTION_DAYS = int(os.getenv("FILE_RETENTION_DAYS", 30))

    ACCESS_CONTROL_SECRET = os.getenv("ACCESS_CONTROL_SECRET", "default_secret")
    ACCESS_TOKEN_EXPIRY = int(os.getenv("ACCESS_TOKEN_EXPIRY", 3600))
