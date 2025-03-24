import asyncio
import logging
from infrastructure.timescaledb.monitoring.metrics_collector import MetricsCollector
from infrastructure.timescaledb.monitoring.slow_query_analyzer import SlowQueryAnalyzer
from infrastructure.timescaledb.monitoring.health_check import HealthCheck
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage
from infrastructure.timescaledb.config.connection_pool import ConnectionPool
from infrastructure.timescaledb.config.settings import TimescaleDBConfig

logger = logging.getLogger(__name__)


async def analyze_performance():
    """
    اجرای فرآیند تحلیل عملکرد پایگاه داده
    """
    logger.info("📊 شروع تحلیل عملکرد پایگاه داده...")

    # مقداردهی اولیه ماژول TimescaleDB
    config = TimescaleDBConfig()
    connection_pool = ConnectionPool(config)
    await connection_pool.initialize()

    storage = TimescaleDBStorage(connection_pool)
    metrics_collector = MetricsCollector(storage)
    slow_query_analyzer = SlowQueryAnalyzer(storage)
    health_check = HealthCheck(storage)

    try:
        # دریافت متریک‌های پایگاه داده
        db_metrics = await metrics_collector.get_database_metrics()
        logger.info(f"📊 متریک‌های پایگاه داده: {db_metrics}")

        # دریافت کندترین کوئری‌ها
        slow_queries = await slow_query_analyzer.get_slow_queries()
        logger.info(f"🐢 کندترین کوئری‌ها: {slow_queries}")

        # بررسی مصرف منابع پایگاه داده
        resource_usage = await health_check.get_resource_usage()
        logger.info(f"📉 مصرف منابع پایگاه داده: {resource_usage}")

        # بررسی مشکلات بحرانی
        critical_issues = await health_check.check_critical_issues()
        logger.info(f"🚨 مشکلات بحرانی پایگاه داده: {critical_issues}")

        logger.info("✅ تحلیل عملکرد پایگاه داده تکمیل شد.")
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل عملکرد پایگاه داده: {e}")

    await connection_pool.close()

# اجرای اسکریپت
if __name__ == "__main__":
    asyncio.run(analyze_performance())
