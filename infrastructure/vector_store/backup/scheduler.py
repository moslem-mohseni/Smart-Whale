import time
import threading
from ..config.config import config
from .backup_service import BackupService


class BackupScheduler:
    """زمان‌بندی خودکار پشتیبان‌گیری برای Collectionهای Milvus"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or collection_config.DEFAULT_COLLECTION_NAME
        self.backup_service = BackupService(self.collection_name)
        self.interval = int(config.BACKUP_INTERVAL)  # بازه زمانی از `.env`

    def start_scheduler(self):
        """شروع فرآیند پشتیبان‌گیری خودکار"""
        def backup_task():
            while True:
                self.backup_service.create_backup()
                time.sleep(self.interval)

        thread = threading.Thread(target=backup_task, daemon=True)
        thread.start()
        print(f"⏳ پشتیبان‌گیری خودکار برای Collection '{self.collection_name}' هر {self.interval} ثانیه اجرا می‌شود.")
