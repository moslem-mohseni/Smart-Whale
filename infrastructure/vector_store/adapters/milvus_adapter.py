from pymilvus import connections
from ..config.config import config


class MilvusAdapter:
    """مدیریت اتصال به Milvus"""

    def __init__(self):
        self.host = config.MILVUS_HOST
        self.port = config.MILVUS_PORT
        self.user = config.MILVUS_USER
        self.password = config.MILVUS_PASSWORD
        self.connected = False

    def connect(self):
        """برقراری اتصال به Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=str(self.port),
                user=self.user,
                password=self.password
            )
            self.connected = True
            print(f"✅ اتصال به Milvus برقرار شد: {self.host}:{self.port}")
        except Exception as e:
            self.connected = False
            print(f"❌ خطا در اتصال به Milvus: {e}")

    def disconnect(self):
        """قطع اتصال از Milvus"""
        try:
            connections.disconnect("default")
            self.connected = False
            print("🔌 اتصال از Milvus قطع شد.")
        except Exception as e:
            print(f"❌ خطا در قطع اتصال: {e}")

    def is_connected(self) -> bool:
        """بررسی وضعیت اتصال"""
        return self.connected

    def reconnect(self):
        """اتصال مجدد به Milvus در صورت قطعی"""
        self.disconnect()
        self.connect()
