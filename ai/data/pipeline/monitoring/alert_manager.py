import asyncio
import logging
from infrastructure.monitoring.prometheus import PrometheusExporter
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.timescaledb.service.database_service import DatabaseService
from infrastructure.monitoring.alerts import AlertNotifier
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

class AlertManager:
    """
    مدیریت هشدارهای عملکردی و نظارتی در `Pipeline`.
    """

    def __init__(self, alert_thresholds: Dict[str, float] = None, check_interval: int = 10):
        """
        مقداردهی اولیه.

        :param alert_thresholds: مقدارهای آستانه برای هشدارهای `Pipeline`
        :param check_interval: فاصله زمانی بین بررسی‌های هشدار (بر حسب ثانیه)
        """
        self.kafka_service = KafkaService()
        self.database_service = DatabaseService()
        self.prometheus_exporter = PrometheusExporter()
        self.alert_notifier = AlertNotifier()
        self.check_interval = check_interval

        # تعریف مقادیر پیش‌فرض برای آستانه‌های هشدار
        self.alert_thresholds = alert_thresholds or {
            "pipeline_latency": 2.0,  # اگر زمان پردازش بیشتر از ۲ ثانیه باشد، هشدار صادر شود
            "error_rate": 0.05,  # اگر نرخ خطا از ۵٪ بیشتر شود، هشدار صادر شود
            "cpu_usage": 80.0,  # اگر مصرف CPU بالای ۸۰٪ باشد، هشدار صادر شود
            "memory_usage": 75.0  # اگر مصرف حافظه بالای ۷۵٪ باشد، هشدار صادر شود
        }

    async def check_pipeline_health(self) -> None:
        """
        بررسی متریک‌های `Pipeline` و تشخیص مشکلات احتمالی.
        """
        while True:
            logging.info("🔍 بررسی وضعیت `Pipeline` برای هشدارها...")
            metrics = await self.database_service.get_latest_metrics("pipeline_metrics")

            if not metrics:
                logging.warning("⚠️ هیچ متریک جدیدی برای بررسی وجود ندارد.")
                await asyncio.sleep(self.check_interval)
                continue

            alerts = []

            for metric, value in metrics.items():
                threshold = self.alert_thresholds.get(metric)
                if threshold and value > threshold:
                    alerts.append(f"🚨 {metric} مقدار {value} دارد که از آستانه {threshold} بالاتر است!")

            if alerts:
                await self.trigger_alerts(alerts)

            await asyncio.sleep(self.check_interval)

    async def trigger_alerts(self, alerts: list) -> None:
        """
        ایجاد و ارسال هشدارها به `Kafka`، `Prometheus` و سیستم‌های اطلاع‌رسانی.

        :param alerts: لیستی از پیام‌های هشدار
        """
        alert_message = "\n".join(alerts)
        logging.warning(f"⚠️ هشدارهای شناسایی‌شده:\n{alert_message}")

        # ارسال هشدارها به Kafka
        await self.kafka_service.send_message({"topic": "pipeline_alerts", "content": {"alerts": alerts}})
        logging.info("📢 هشدارها در `Kafka` منتشر شدند.")

        # ارسال هشدارها به Prometheus
        await self.prometheus_exporter.export({"pipeline_alerts": len(alerts)})
        logging.info("✅ هشدارها به `Prometheus` ارسال شدند.")

        # ارسال هشدار به Slack، Email یا سیستم‌های دیگر
        await self.alert_notifier.send_alert(alert_message)
        logging.info("📩 هشدارها به سیستم‌های اطلاع‌رسانی ارسال شدند.")

# مقداردهی اولیه و راه‌اندازی `AlertManager`
async def start_alert_manager():
    alert_manager = AlertManager()
    await alert_manager.check_pipeline_health()

asyncio.create_task(start_alert_manager())
