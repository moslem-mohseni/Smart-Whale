# ai/core/orchestration/alert_manager.py
"""
مدیریت هشدارها و شناسایی شرایط غیرعادی

این ماژول مسئول نظارت بر متریک‌های سیستم و صدور هشدار در صورت مشاهده شرایط غیرعادی است.
همچنین قابلیت ارسال اعلان به کانال‌های مختلف (ایمیل، پیامک، وبهوک و غیره) را دارد.
این سیستم از قوانین قابل پیکربندی برای تشخیص شرایط هشدار استفاده می‌کند.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
from dataclasses import dataclass
import smtplib
import aiohttp
from email.mime.text import MIMEText
import uuid

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """قانون تشخیص هشدار"""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: float
    severity: str  # 'critical', 'warning', 'info'
    description: str
    cooldown: int = 300  # مدت زمان انتظار بین هشدارهای مشابه (ثانیه)


@dataclass
class Alert:
    """ساختار یک هشدار"""
    alert_id: str
    rule_id: str
    severity: str
    message: str
    metric_value: float
    threshold: float
    created_at: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    notified: bool = False
    acknowledgment: Optional[Dict[str, Any]] = None


@dataclass
class NotificationConfig:
    """تنظیمات اعلان‌ها"""
    email_config: Optional[Dict[str, str]] = None
    slack_webhook: Optional[str] = None
    telegram_config: Optional[Dict[str, str]] = None
    custom_webhook: Optional[str] = None


class AlertManager:
    """مدیریت هشدارها و اعلان‌ها"""

    def __init__(self, notification_config: NotificationConfig):
        """
        راه‌اندازی مدیر هشدارها

        Args:
            notification_config: تنظیمات مربوط به ارسال اعلان‌ها
        """
        self.notification_config = notification_config
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_notification: Dict[str, datetime] = {}

        # مسیر ذخیره‌سازی هشدارها
        self.alerts_dir = Path("data/alerts")
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        # ثبت‌کننده‌های اعلان
        self.notification_handlers: List[Callable[[Alert], None]] = []

        # قوانین پیش‌فرض
        self._setup_default_rules()

    def _setup_default_rules(self):
        """تنظیم قوانین پیش‌فرض هشدار"""
        default_rules = [
            AlertRule(
                rule_id="high_error_rate",
                name="نرخ خطای بالا",
                metric_name="error_rate",
                condition="gt",
                threshold=0.1,  # 10%
                severity="critical",
                description="نرخ خطای سیستم از حد مجاز بیشتر شده است"
            ),
            AlertRule(
                rule_id="high_latency",
                name="تأخیر بالا",
                metric_name="response_time",
                condition="gt",
                threshold=2.0,  # 2 seconds
                severity="warning",
                description="زمان پاسخ‌دهی سیستم بیش از حد طولانی شده است"
            ),
            AlertRule(
                rule_id="low_accuracy",
                name="دقت پایین",
                metric_name="model_accuracy",
                condition="lt",
                threshold=0.8,  # 80%
                severity="warning",
                description="دقت مدل کمتر از حد قابل قبول است"
            ),
            AlertRule(
                rule_id="high_memory",
                name="مصرف حافظه بالا",
                metric_name="memory_usage",
                condition="gt",
                threshold=0.9,  # 90%
                severity="critical",
                description="مصرف حافظه به حد بحرانی رسیده است"
            )
        ]

        for rule in default_rules:
            self.add_rule(rule)

    def add_rule(self, rule: AlertRule):
        """
        افزودن یک قانون جدید

        Args:
            rule: قانون هشدار جدید
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"Added new alert rule: {rule.name}")

    async def check_metrics(self, metrics: Dict[str, Any]):
        """
        بررسی متریک‌ها و صدور هشدار در صورت نیاز

        Args:
            metrics: دیکشنری متریک‌های سیستم
        """
        for rule in self.rules.values():
            try:
                metric_value = self._get_metric_value(metrics, rule.metric_name)
                if metric_value is not None and self._check_condition(
                        metric_value, rule.condition, rule.threshold):

                    await self._create_alert(rule, metric_value)
                else:
                    # بررسی رفع هشدارهای فعال مربوط به این قانون
                    await self._resolve_alerts(rule.rule_id)

            except Exception as e:
                logger.error(f"Error checking rule {rule.name}: {str(e)}")

    def _get_metric_value(self, metrics: Dict[str, Any],
                          metric_name: str) -> Optional[float]:
        """استخراج مقدار متریک از ساختار داده"""
        try:
            # پشتیبانی از متریک‌های تودرتو با نقطه
            parts = metric_name.split('.')
            value = metrics
            for part in parts:
                value = value[part]
            return float(value)
        except (KeyError, ValueError, TypeError):
            return None

    def _check_condition(self, value: float, condition: str,
                         threshold: float) -> bool:
        """بررسی شرط هشدار"""
        if condition == 'gt':
            return value > threshold
        elif condition == 'lt':
            return value < threshold
        elif condition == 'eq':
            return abs(value - threshold) < 1e-6
        elif condition == 'ne':
            return abs(value - threshold) >= 1e-6
        return False

    async def _create_alert(self, rule: AlertRule, metric_value: float):
        """ایجاد هشدار جدید"""
        # بررسی cooldown
        last_alert_time = self.last_notification.get(rule.rule_id)
        if last_alert_time and (datetime.now() - last_alert_time).total_seconds() < rule.cooldown:
            return

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            severity=rule.severity,
            message=rule.description,
            metric_value=metric_value,
            threshold=rule.threshold,
            created_at=datetime.now()
        )

        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        await self._save_alert(alert)
        await self._send_notifications(alert)

        self.last_notification[rule.rule_id] = datetime.now()
        logger.warning(f"Created new alert: {rule.name} ({alert.alert_id})")

    async def _resolve_alerts(self, rule_id: str):
        """رفع هشدارهای مربوط به یک قانون"""
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                await self._save_alert(alert)
                logger.info(f"Resolved alert: {alert.alert_id}")

    async def _save_alert(self, alert: Alert):
        """ذخیره هشدار در فایل"""
        alert_file = self.alerts_dir / f"{alert.created_at.strftime('%Y%m')}_alerts.jsonl"
        try:
            with open(alert_file, 'a', encoding='utf-8') as f:
                alert_dict = {
                    k: str(v) if isinstance(v, datetime) else v
                    for k, v in alert.__dict__.items()
                }
                f.write(json.dumps(alert_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to save alert: {str(e)}")

    async def _send_notifications(self, alert: Alert):
        """ارسال اعلان‌ها"""
        tasks = []

        # ارسال ایمیل
        if self.notification_config.email_config:
            tasks.append(self._send_email_notification(alert))

        # ارسال به Slack
        if self.notification_config.slack_webhook:
            tasks.append(self._send_slack_notification(alert))

        # ارسال به Telegram
        if self.notification_config.telegram_config:
            tasks.append(self._send_telegram_notification(alert))

        # ارسال به webhook سفارشی
        if self.notification_config.custom_webhook:
            tasks.append(self._send_webhook_notification(alert))

        # اجرای همه task های ارسال به صورت همزمان
        await asyncio.gather(*tasks, return_exceptions=True)

        # فراخوانی handler های سفارشی
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {str(e)}")

    async def _send_email_notification(self, alert: Alert):
        """ارسال اعلان از طریق ایمیل"""
        if not self.notification_config.email_config:
            return

        try:
            config = self.notification_config.email_config
            message = MIMEText(self._create_alert_message(alert))
            message['Subject'] = f"[{alert.severity.upper()}] {self.rules[alert.rule_id].name}"
            message['From'] = config['from']
            message['To'] = config['to']

            with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
                if config.get('smtp_user') and config.get('smtp_password'):
                    server.login(config['smtp_user'], config['smtp_password'])
                server.send_message(message)

            logger.info(f"Sent email notification for alert {alert.alert_id}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """افزودن handler سفارشی برای اعلان‌ها"""
        self.notification_handlers.append(handler)

    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """دریافت هشدارهای فعال"""
        alerts = list(filter(lambda x: not x.resolved, self.active_alerts.values()))
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)

    def acknowledge_alert(self, alert_id: str, acknowledgment: Dict[str, Any]) -> bool:
        """تأیید دریافت و بررسی هشدار"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledgment = {
                **acknowledgment,
                'timestamp': datetime.now().isoformat()
            }
            return True
        return False