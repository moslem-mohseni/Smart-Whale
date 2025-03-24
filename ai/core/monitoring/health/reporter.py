import logging
import requests
import smtplib
from email.mime.text import MIMEText
from ai.core.monitoring.health.checker import HealthChecker


class HealthReporter:
    def __init__(self, health_checker: HealthChecker, slack_webhook=None, email_config=None, logstash_url=None):
        """
        ارسال گزارش‌های سلامت سیستم به Slack، Email و Logstash
        :param health_checker: نمونه‌ای از HealthChecker برای دریافت داده‌های سلامت
        :param slack_webhook: لینک Webhook برای ارسال گزارش‌ها به Slack
        :param email_config: تنظیمات ایمیل (دیکشنری شامل `smtp_server`, `port`, `sender_email`, `receiver_email`, `password`)
        :param logstash_url: لینک Logstash برای ارسال گزارش‌ها
        """
        self.health_checker = health_checker
        self.slack_webhook = slack_webhook
        self.email_config = email_config
        self.logstash_url = logstash_url
        self.logger = logging.getLogger("HealthReporter")

    def generate_report(self):
        """ تولید گزارش سلامت سیستم """
        report = self.health_checker.run_health_checks()
        report_text = "\n".join([f"{key}: {'✅ سالم' if value else '⚠️ مشکل'}" for key, value in report.items()])
        return report_text

    def send_to_slack(self, report):
        """ ارسال گزارش سلامت به Slack """
        if self.slack_webhook:
            payload = {"text": f"📢 گزارش سلامت سیستم:\n{report}"}
            try:
                response = requests.post(self.slack_webhook, json=payload)
                if response.status_code == 200:
                    self.logger.info("✅ گزارش سلامت به Slack ارسال شد.")
                else:
                    self.logger.error(f"❌ خطا در ارسال به Slack: {response.status_code}")
            except Exception as e:
                self.logger.error(f"❌ خطا در ارسال به Slack: {e}")

    def send_email(self, report):
        """ ارسال گزارش سلامت از طریق ایمیل """
        if self.email_config:
            msg = MIMEText(report)
            msg["Subject"] = "📢 گزارش سلامت سیستم"
            msg["From"] = self.email_config["sender_email"]
            msg["To"] = self.email_config["receiver_email"]

            try:
                with smtplib.SMTP(self.email_config["smtp_server"], self.email_config["port"]) as server:
                    server.starttls()
                    server.login(self.email_config["sender_email"], self.email_config["password"])
                    server.sendmail(self.email_config["sender_email"], self.email_config["receiver_email"], msg.as_string())
                self.logger.info("✅ گزارش سلامت از طریق ایمیل ارسال شد.")
            except Exception as e:
                self.logger.error(f"❌ خطا در ارسال ایمیل: {e}")

    def send_to_logstash(self, report):
        """ ارسال گزارش سلامت به Logstash """
        if self.logstash_url:
            try:
                response = requests.post(self.logstash_url, json={"health_report": report})
                if response.status_code == 200:
                    self.logger.info("✅ گزارش سلامت به Logstash ارسال شد.")
                else:
                    self.logger.error(f"❌ خطا در ارسال به Logstash: {response.status_code}")
            except Exception as e:
                self.logger.error(f"❌ خطا در ارسال به Logstash: {e}")

    def report_health(self):
        """ اجرای ارسال گزارش سلامت به تمامی کانال‌های موجود """
        report = self.generate_report()
        self.logger.info(f"📢 گزارش سلامت سیستم:\n{report}")
        self.send_to_slack(report)
        self.send_email(report)
        self.send_to_logstash(report)
