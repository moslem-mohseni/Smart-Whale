import logging
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from prometheus_client import Gauge


class AlertVisualizer:
    def __init__(self, prometheus_url=None):
        """
        نمایش هشدارهای سیستم به‌صورت گرافیکی و متنی
        :param prometheus_url: آدرس Prometheus برای دریافت هشدارها
        """
        self.prometheus_url = prometheus_url
        self.logger = logging.getLogger("AlertVisualizer")

        # متریک شمارش هشدارهای فعال در سیستم
        self.active_alerts = Gauge("active_alerts_count", "Number of currently active alerts")

    def fetch_alerts_from_prometheus(self):
        """ دریافت هشدارهای فعال از Prometheus """
        if not self.prometheus_url:
            self.logger.warning("❌ Prometheus URL تنظیم نشده است.")
            return None

        query_url = f"{self.prometheus_url}/api/v1/query?query=ALERTS"
        try:
            response = requests.get(query_url)
            if response.status_code == 200:
                return response.json().get("data", {}).get("result", [])
            else:
                self.logger.error(f"⚠️ خطا در دریافت هشدارها از Prometheus: {response.status_code}")
        except Exception as e:
            self.logger.error(f"❌ خطا در برقراری ارتباط با Prometheus: {e}")

        return None

    def generate_alert_dashboard(self, alert_data):
        """ تولید داشبورد بصری از هشدارها با Matplotlib """
        if not alert_data:
            self.logger.warning("⚠️ هیچ هشداری برای نمایش وجود ندارد.")
            return

        alert_labels = [alert["labels"]["alertname"] for alert in alert_data]
        alert_counts = [1] * len(alert_labels)  # هر هشدار به‌عنوان یک واحد شمارش می‌شود

        plt.figure(figsize=(8, 5))
        plt.bar(alert_labels, alert_counts, color="red")
        plt.xlabel("Alerts")
        plt.ylabel("Occurrences")
        plt.title(f"Active System Alerts - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        plt.xticks(rotation=45, ha="right")
        plt.savefig("alert_dashboard.png")
        plt.close()

        self.logger.info("📊 داشبورد هشدارها به‌صورت PNG ذخیره شد.")

    def visualize_alerts(self):
        """ اجرای دریافت هشدارها و تولید داشبورد گرافیکی """
        if self.prometheus_url:
            self.logger.info("📡 دریافت داده‌های هشدار از Prometheus...")
            alert_data = self.fetch_alerts_from_prometheus()
            if alert_data:
                self.logger.info(f"✅ تعداد هشدارهای فعال: {len(alert_data)}")
                self.active_alerts.set(len(alert_data))
                self.generate_alert_dashboard(alert_data)
            else:
                self.logger.info("✅ هیچ هشداری در سیستم فعال نیست.")
