import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Optional

# بارگذاری متغیرهای محیطی از `.env`
load_dotenv()

@dataclass
class KafkaConfig:
    """
    تنظیمات اتصال به Kafka
    """
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    client_id: str = os.getenv("KAFKA_CLIENT_ID", "kafka_client")
    group_id: Optional[str] = os.getenv("KAFKA_GROUP_ID", "default_group")
    security_protocol: str = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    sasl_mechanism: Optional[str] = os.getenv("KAFKA_SASL_MECHANISM")
    sasl_username: Optional[str] = os.getenv("KAFKA_SASL_USERNAME")
    sasl_password: Optional[str] = os.getenv("KAFKA_SASL_PASSWORD")

    def get_producer_config(self) -> dict:
        """تنظیمات Kafka برای تولیدکننده"""
        config = {
            "bootstrap.servers": ",".join(self.bootstrap_servers),
            "client.id": self.client_id,
        }
        return self._add_security_config(config)

    def get_consumer_config(self) -> dict:
        """تنظیمات Kafka برای مصرف‌کننده"""
        config = {
            "bootstrap.servers": ",".join(self.bootstrap_servers),
            "group.id": self.group_id,
            "auto.offset.reset": "earliest"
        }
        return self._add_security_config(config)

    def _add_security_config(self, config: dict) -> dict:
        """اضافه کردن تنظیمات امنیتی (در صورت وجود)"""
        if self.security_protocol != "PLAINTEXT":
            config["security.protocol"] = self.security_protocol
            if self.sasl_mechanism:
                config["sasl.mechanism"] = self.sasl_mechanism
                config["sasl.username"] = self.sasl_username
                config["sasl.password"] = self.sasl_password
        return config


@dataclass
class RedisConfig:
    """
    تنظیمات اتصال به Redis
    """
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", 6379))
    db: int = int(os.getenv("REDIS_DB", 0))
    password: Optional[str] = os.getenv("REDIS_PASSWORD", None)

    def get_redis_url(self) -> str:
        """برگرداندن آدرس Redis به صورت رشته مناسب برای اتصال"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
