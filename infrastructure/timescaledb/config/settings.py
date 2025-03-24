import os
from dataclasses import dataclass
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی از فایل .env
load_dotenv()


@dataclass
class TimescaleDBConfig:
    """کلاس تنظیمات اتصال به TimescaleDB"""

    host: str = os.getenv("TIMESCALEDB_HOST", "localhost")
    port: int = int(os.getenv("TIMESCALEDB_PORT", 5432))
    database: str = os.getenv("TIMESCALEDB_DATABASE", "timeseries_db")
    user: str = os.getenv("TIMESCALEDB_USER", "db_user")
    password: str = os.getenv("TIMESCALEDB_PASSWORD", "db_password")

    min_connections: int = int(os.getenv("TIMESCALEDB_MIN_CONNECTIONS", 2))
    max_connections: int = int(os.getenv("TIMESCALEDB_MAX_CONNECTIONS", 20))
    connection_timeout: int = int(os.getenv("TIMESCALEDB_CONNECTION_TIMEOUT", 30))

    # تنظیمات Replication برای Read/Write Split
    read_replica: str = os.getenv("TIMESCALEDB_READ_REPLICA", None)

    def get_connection_string(self) -> str:
        """ساخت رشته اتصال برای TimescaleDB"""
        return (f"postgresql://{self.user}:{self.password}@{self.host}:"
                f"{self.port}/{self.database}")

    def get_read_replica_connection(self) -> str:
        """رشته اتصال برای Read Replica (در صورت وجود)"""
        if self.read_replica:
            return f"postgresql://{self.user}:{self.password}@{self.read_replica}:{self.port}/{self.database}"
        return self.get_connection_string()
