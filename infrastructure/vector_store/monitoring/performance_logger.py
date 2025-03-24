# infrastructure/vector_store/monitoring/performance_logger.py

import time
import logging
from .metrics import metrics

# تنظیمات اولیه لاگ‌ها
logging.basicConfig(filename="performance.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PerformanceLogger:
    """ثبت متریک‌های عملکردی برای Vector Store با Prometheus و لاگ"""

    @staticmethod
    def log_operation(operation_type: str):
        """دکوراتور عمومی برای ثبت تأخیر و لاگ هر عملیات"""

        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time

                # ثبت متریک مناسب بر اساس نوع عملیات
                if operation_type.lower() == "insert":
                    metrics.record_insert_latency(elapsed_time)
                elif operation_type.lower() == "delete":
                    metrics.record_delete_latency(elapsed_time)
                elif operation_type.lower() == "search":
                    metrics.record_search_latency(elapsed_time)

                # ثبت در لاگ
                logger.info(f"⏱️ {operation_type} زمان اجرا: {elapsed_time:.4f} ثانیه")
                print(f"⏱️ {operation_type} زمان اجرا: {elapsed_time:.4f} ثانیه")

                return result

            return wrapper

        return decorator


# دکوراتورهای خاص برای هر عملیات
log_insert = PerformanceLogger.log_operation("Insert")
log_delete = PerformanceLogger.log_operation("Delete")
log_search = PerformanceLogger.log_operation("Search")
