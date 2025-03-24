import logging
import smtplib
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)

class KafkaAlerts:
    """
    سیستم هشداردهی برای مشکلات Kafka
    """

    def __init__(self, alert_email: Optional[str] = None, smtp_server: Optional[str] = None):
        """
        مقداردهی اولیه سیستم هشدار

        :param alert_email: ایمیل مقصد برای ارسال هشدارها
        :param smtp_server: سرور SMTP برای ارسال ایمیل
        """
        self.alert_email = alert_email
        self.smtp_server = smtp_server

    def send_alert(self, message: str, level: str = "WARNING"):
        """
        ارسال هشدار از طریق Logger و در صورت تنظیم، از طریق ایمیل

        :param message: متن هشدار
        :param level: سطح هشدار (WARNING, CRITICAL, EMERGENCY)
        """
        log_message = f"[Kafka Alert] [{level}] {message}"

        if level == "CRITICAL":
            logger.critical(log_message)
        elif level == "EMERGENCY":
            logger.error(log_message)
        else:
            logger.warning(log_message)

        if self.alert_email and self.smtp_server:
            self._send_email_alert(message, level)

    def _send_email_alert(self, message: str, level: str):
        """
        ارسال هشدار از طریق ایمیل
        """
        try:
            msg = MIMEText(f"Kafka Alert: {level}\n\n{message}")
            msg["Subject"] = f"Kafka Alert: {level}"
            msg["From"] = "noreply@kafka-monitor.com"
            msg["To"] = self.alert_email

            with smtplib.SMTP(self.smtp_server) as server:
                server.sendmail("noreply@kafka-monitor.com", [self.alert_email], msg.as_string())

            logger.info(f"Email alert sent to {self.alert_email}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
