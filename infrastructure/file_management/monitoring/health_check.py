import os
from prometheus_client import Gauge
from infrastructure.file_management.adapters.minio_adapter import MinioAdapter


class HealthCheck:
    """
    بررسی سلامت سیستم و ارتباط با MinIO
    """

    def __init__(self):
        self.minio_adapter = MinioAdapter()
        self.system_health_gauge = Gauge("file_management_health_status", "Health status of file management system")

    def check_minio_connection(self) -> bool:
        """بررسی ارتباط با MinIO"""
        try:
            if self.minio_adapter.client.bucket_exists(os.getenv("MINIO_BUCKET_NAME", "default")):
                self.system_health_gauge.set(1)  # Healthy
                return True
        except Exception:
            self.system_health_gauge.set(0)  # Unhealthy
            return False
        return False