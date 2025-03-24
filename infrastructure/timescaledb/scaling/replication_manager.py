import logging
import os
from typing import Optional, Dict, Any
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

logger = logging.getLogger(__name__)


class ReplicationManager:
    """مدیریت Replication در TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
        """
        self.storage = storage
        self.replication_lag_threshold = int(os.getenv("REPLICATION_LAG_THRESHOLD", 5))  # مقدار مجاز تأخیر در ثانیه

    async def check_replication_status(self) -> Optional[Dict[str, Any]]:
        """
        بررسی وضعیت Replication در پایگاه داده

        Returns:
            Optional[Dict[str, Any]]: اطلاعات نودهای Replication یا None در صورت بروز خطا
        """
        query = """
            SELECT application_name, client_addr, state, sync_priority, sync_state, 
                   pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS replication_lag
            FROM pg_stat_replication;
        """
        try:
            result = await self.storage.execute_query(query)
            if result:
                logger.info("📡 وضعیت Replication بررسی شد.")
                return result
            else:
                logger.warning("⚠️ هیچ نود Replication متصل نیست.")
                return None
        except Exception as e:
            logger.error(f"❌ خطا در بررسی وضعیت Replication: {e}")
            return None

    async def promote_standby(self):
        """
        ارتقای نود Standby به نود اصلی در صورت Failover

        Raises:
            Exception: در صورت بروز خطا در ارتقا
        """
        try:
            logger.warning("🚨 انجام Failover: ارتقای نود Standby به Master...")
            await self.storage.execute_query("SELECT pg_promote();")
            logger.info("✅ نود Standby به Master ارتقا یافت.")
        except Exception as e:
            logger.error(f"❌ خطا در ارتقای Standby به Master: {e}")
            raise

    async def monitor_replication_lag(self):
        """
        بررسی تأخیر Replication و هشدار در صورت عبور از آستانه مجاز
        """
        try:
            status = await self.check_replication_status()
            if status:
                for node in status:
                    replication_lag = node.get("replication_lag", 0)
                    if replication_lag > self.replication_lag_threshold:
                        logger.warning(f"⚠️ تأخیر Replication برای {node['application_name']} بیش از حد مجاز است: {replication_lag} ثانیه")
        except Exception as e:
            logger.error(f"❌ خطا در بررسی تأخیر Replication: {e}")
