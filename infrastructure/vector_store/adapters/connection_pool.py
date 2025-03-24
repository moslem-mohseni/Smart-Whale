import queue
from pymilvus import connections
from ..config.config import config


def create_connection():
    """ایجاد یک اتصال جدید به Milvus"""
    conn = connections.connect(
        alias="pool_connection",
        host=config.MILVUS_HOST,
        port=str(config.MILVUS_PORT),
        user=config.MILVUS_USER,
        password=config.MILVUS_PASSWORD
    )
    return conn


class ConnectionPool:
    """مدیریت Connection Pool برای Milvus"""

    def __init__(self, max_connections: int = None):
        """
        مقداردهی اولیه Connection Pool
        :param max_connections: حداکثر تعداد اتصال‌های همزمان (خوانده شده از .env)
        """
        self.max_connections = max_connections or int(config.MILVUS_MAX_CONNECTIONS)
        self.pool = queue.Queue(maxsize=self.max_connections)

        # ایجاد اتصال‌های اولیه در Pool
        for _ in range(self.max_connections):
            conn = create_connection()
            self.pool.put(conn)

    def get_connection(self):
        """دریافت یک اتصال از Pool"""
        if self.pool.empty():
            print("⚠️ هشدار: همه‌ی اتصالات در حال استفاده هستند. لطفاً منتظر بمانید...")
        return self.pool.get()

    def release_connection(self, conn):
        """بازگرداندن اتصال به Pool برای استفاده مجدد"""
        self.pool.put(conn)

    def close_all_connections(self):
        """بستن تمام اتصالات موجود در Pool"""
        while not self.pool.empty():
            conn = self.pool.get()
            connections.disconnect("pool_connection")

        print("🔌 تمامی اتصالات به Milvus بسته شدند.")
