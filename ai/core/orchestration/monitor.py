# ai/core/orchestration/monitor.py
"""
سیستم مانیتورینگ و نظارت بر عملکرد

این ماژول مسئول هماهنگی بین بخش‌های مختلف مانیتورینگ (جمع‌آوری متریک‌ها و مدیریت هشدارها)
و ارائه یک رابط یکپارچه برای نظارت بر کل سیستم است.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

from .metrics_collector import MetricsCollector, MetricConfig
from .alert_manager import AlertManager, AlertRule, NotificationConfig

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """تنظیمات جامع مانیتورینگ"""
    metric_config: MetricConfig
    notification_config: NotificationConfig
    check_interval: int = 60  # فاصله بررسی‌های دوره‌ای (ثانیه)
    status_retention_days: int = 30  # مدت نگهداری وضعیت‌ها
    enable_prometheus: bool = True  # فعال‌سازی پشتیبانی از Prometheus


class SystemMonitor:
    """مدیریت یکپارچه مانیتورینگ سیستم"""

    def __init__(self, config: MonitoringConfig):
        """
        راه‌اندازی سیستم مانیتورینگ

        Args:
            config: تنظیمات مانیتورینگ
        """
        self.config = config

        # راه‌اندازی زیرسیستم‌ها
        self.metrics_collector = MetricsCollector(config.metric_config)
        self.alert_manager = AlertManager(config.notification_config)

        self._monitor_task: Optional[asyncio.Task] = None
        self._should_stop = False

        # ثبت observer برای تغییرات متریک‌ها
        self._metric_observers: List[Callable[[Dict[str, Any]], None]] = []

    async def start(self):
        """شروع فرآیند مانیتورینگ"""
        if not self._monitor_task:
            self._should_stop = False
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            logger.info("System monitoring started")

    async def stop(self):
        """توقف فرآیند مانیتورینگ"""
        if self._monitor_task:
            self._should_stop = True
            await self._monitor_task
            self._monitor_task = None

            # پاکسازی منابع زیرسیستم‌ها
            await self.metrics_collector.cleanup()
            await self.alert_manager.cleanup()

            logger.info("System monitoring stopped")

    async def _monitoring_loop(self):
        """حلقه اصلی مانیتورینگ"""
        while not self._should_stop:
            try:
                # جمع‌آوری متریک‌های سیستم
                metrics = await self.metrics_collector.collect_all_metrics()

                # اطلاع‌رسانی به observer ها
                for observer in self._metric_observers:
                    try:
                        observer(metrics)
                    except Exception as e:
                        logger.error(f"Metric observer failed: {str(e)}")

                # بررسی شرایط هشدار
                await self.alert_manager.check_metrics(metrics)

                # انتظار تا بررسی بعدی
                await asyncio.sleep(self.config.check_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(60)  # انتظار قبل از تلاش مجدد

    def add_metric_observer(self, observer: Callable[[Dict[str, Any]], None]):
        """
        افزودن observer برای متریک‌ها

        Args:
            observer: تابعی که با هر بروزرسانی متریک‌ها فراخوانی می‌شود
        """
        self._metric_observers.append(observer)

    def add_alert_rule(self, rule: AlertRule):
        """
        افزودن قانون هشدار جدید

        Args:
            rule: قانون هشدار جدید
        """
        self.alert_manager.add_rule(rule)

    def add_notification_handler(self, handler: Callable):
        """
        افزودن handler برای اعلان‌ها

        Args:
            handler: تابع پردازش‌کننده اعلان‌ها
        """
        self.alert_manager.add_notification_handler(handler)

    async def get_system_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت کلی سیستم

        Returns:
            دیکشنری حاوی وضعیت سیستم
        """
        metrics = await self.metrics_collector.get_aggregated_metrics()
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            'status': 'healthy' if not active_alerts else 'warning',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'active_alerts': [
                {
                    'id': alert.alert_id,
                    'severity': alert.severity,
                    'message': alert.message,
                    'created_at': alert.created_at.isoformat()
                }
                for alert in active_alerts
            ],
            'monitoring_info': {
                'uptime': self.get_uptime(),
                'last_check': datetime.now().isoformat(),
                'observers_count': len(self._metric_observers)
            }
        }

    def get_uptime(self) -> float:
        """محاسبه مدت زمان فعال بودن سیستم"""
        if not hasattr(self, '_start_time'):
            self._start_time = datetime.now()
        return (datetime.now() - self._start_time).total_seconds()

    def get_metrics_collectors(self) -> List[str]:
        """دریافت لیست جمع‌کننده‌های متریک فعال"""
        return self.metrics_collector.get_active_collectors()

    def get_alert_rules(self) -> List[AlertRule]:
        """دریافت لیست قوانین هشدار"""
        return list(self.alert_manager.rules.values())