import time
import psutil
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class KafkaMetrics:
    """
    مانیتورینگ متریک‌های کلیدی Kafka
    """

    def __init__(self):
        self.message_sent_count = 0
        self.message_received_count = 0
        self.start_time = time.time()

    def record_message_sent(self):
        """ثبت یک پیام ارسال شده"""
        self.message_sent_count += 1

    def record_message_received(self):
        """ثبت یک پیام دریافت شده"""
        self.message_received_count += 1

    def get_throughput(self) -> Dict[str, float]:
        """
        نرخ ارسال و دریافت پیام‌ها بر حسب پیام در ثانیه

        :return: دیکشنری شامل نرخ ارسال و دریافت پیام‌ها
        """
        elapsed_time = time.time() - self.start_time
        return {
            "messages_sent_per_sec": self.message_sent_count / elapsed_time if elapsed_time > 0 else 0,
            "messages_received_per_sec": self.message_received_count / elapsed_time if elapsed_time > 0 else 0
        }

    def get_system_metrics(self) -> Dict[str, float]:
        """
        جمع‌آوری متریک‌های مصرف منابع سیستم

        :return: دیکشنری شامل مصرف CPU و حافظه
        """
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent
        }

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        دریافت تمام متریک‌های سیستم

        :return: دیکشنری شامل نرخ پیام‌ها و مصرف منابع
        """
        return {
            "kafka_throughput": self.get_throughput(),
            "system_metrics": self.get_system_metrics()
        }
