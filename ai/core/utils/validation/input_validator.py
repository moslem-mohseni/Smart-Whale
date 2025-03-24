import re
import uuid
import logging
from urllib.parse import urlparse
from ipaddress import ip_address, IPv4Address, IPv6Address

class InputValidator:
    def __init__(self):
        """
        ابزار اعتبارسنجی انواع ورودی‌های متنی و عددی
        """
        self.logger = logging.getLogger("InputValidator")

    def validate_string(self, value: str, min_length=1, max_length=255) -> bool:
        """ بررسی صحت ورودی متنی """
        if min_length <= len(value) <= max_length:
            return True
        self.logger.warning(f"❌ طول رشته نامعتبر: {value}")
        return False

    def validate_integer(self, value: int, min_value=None, max_value=None) -> bool:
        """ بررسی صحت عدد صحیح """
        if (min_value is None or value >= min_value) and (max_value is None or value <= max_value):
            return True
        self.logger.warning(f"❌ مقدار عدد صحیح نامعتبر: {value}")
        return False

    def validate_float(self, value: float, min_value=None, max_value=None) -> bool:
        """ بررسی صحت عدد اعشاری """
        if (min_value is None or value >= min_value) and (max_value is None or value <= max_value):
            return True
        self.logger.warning(f"❌ مقدار عدد اعشاری نامعتبر: {value}")
        return False

    def validate_uuid(self, value: str) -> bool:
        """ بررسی صحت UUID """
        try:
            uuid.UUID(value)
            return True
        except ValueError:
            self.logger.warning(f"❌ مقدار UUID نامعتبر: {value}")
            return False

    def validate_ip(self, value: str) -> bool:
        """ بررسی صحت آدرس IP (IPv4 یا IPv6) """
        try:
            return isinstance(ip_address(value), (IPv4Address, IPv6Address))
        except ValueError:
            self.logger.warning(f"❌ مقدار آدرس IP نامعتبر: {value}")
            return False

    def validate_url(self, value: str) -> bool:
        """ بررسی صحت URL """
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            return True
        self.logger.warning(f"❌ مقدار URL نامعتبر: {value}")
        return False
