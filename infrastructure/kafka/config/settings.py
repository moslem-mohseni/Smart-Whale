# infrastructure/kafka/config/settings.py
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KafkaConfig:
    """
    تنظیمات اتصال به کافکا

    این کلاس تمام پارامترهای مورد نیاز برای اتصال به کافکا را نگهداری می‌کند.
    """
    bootstrap_servers: List[str]
    client_id: str
    group_id: Optional[str] = None
    security_protocol: str = 'PLAINTEXT'
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

    def get_producer_config(self) -> dict:
        """تنظیمات مورد نیاز برای تولیدکننده"""
        config = {
            'bootstrap.servers': ','.join(self.bootstrap_servers),
            'client.id': self.client_id,
        }
        return self._add_security_config(config)

    def get_consumer_config(self) -> dict:
        """تنظیمات مورد نیاز برای مصرف‌کننده"""
        if not self.group_id:
            raise ValueError("group_id is required for consumer")

        config = {
            'bootstrap.servers': ','.join(self.bootstrap_servers),
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest'
        }
        return self._add_security_config(config)

    def _add_security_config(self, config: dict) -> dict:
        """اضافه کردن تنظیمات امنیتی"""
        if self.security_protocol != 'PLAINTEXT':
            config['security.protocol'] = self.security_protocol
            if self.sasl_mechanism:
                config['sasl.mechanism'] = self.sasl_mechanism
                config['sasl.username'] = self.sasl_username
                config['sasl.password'] = self.sasl_password
        return config