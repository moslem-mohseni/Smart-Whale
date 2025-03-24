import os
from minio import Minio
from threading import Lock


class MinioConnectionPool:
    """
    مدیریت Connection Pool برای ارتباط با MinIO
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MinioConnectionPool, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """مقداردهی اولیه Connection Pool"""
        self.client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "your_access_key"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "your_secret_key"),
            secure=False
        )

    def get_client(self):
        """دریافت یک نمونه از کلاینت MinIO"""
        return self.client
