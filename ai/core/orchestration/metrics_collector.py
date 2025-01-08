# ai/core/orchestration/metrics_collector.py
"""
جمع‌آوری کننده متریک‌های سیستم

این ماژول مسئول جمع‌آوری، ذخیره و مدیریت متریک‌های مختلف سیستم است. متریک‌ها شامل
اطلاعات عملکردی مدل‌ها، وضعیت سیستم و آمار درخواست‌ها می‌شوند. این کلاس از Prometheus
برای ارائه متریک‌ها به صورت استاندارد استفاده می‌کند.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import prometheus_client as prom
from dataclasses import dataclass
import numpy as np
import psutil
import torch
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """تنظیمات متریک‌ها"""
    history_size: int = 1000  # تعداد نمونه‌های تاریخی برای هر متریک
    aggregation_interval: int = 60  # فاصله تجمیع داده‌ها (ثانیه)
    cleanup_interval: int = 3600  # فاصله پاکسازی داده‌های قدیمی (ثانیه)


class MetricsCollector:
    """جمع‌آوری و مدیریت متریک‌های سیستم"""

    def __init__(self, config: MetricConfig):
        """
        راه‌اندازی جمع‌کننده متریک‌ها

        Args:
            config: تنظیمات مربوط به متریک‌ها
        """
        self.config = config
        self._initialize_prometheus_metrics()
        self._initialize_storage()

    def _initialize_prometheus_metrics(self):
        """راه‌اندازی متریک‌های Prometheus"""
        # متریک‌های مربوط به درخواست‌ها
        self.request_counter = prom.Counter(
            'ai_requests_total',
            'Total number of AI requests',
            ['model_type', 'status']
        )
        self.request_duration = prom.Histogram(
            'ai_request_duration_seconds',
            'Request duration in seconds',
            ['model_type'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )

        # متریک‌های مربوط به مدل
        self.model_accuracy = prom.Gauge(
            'ai_model_accuracy',
            'Model accuracy',
            ['model_type', 'version']
        )
        self.model_latency = prom.Gauge(
            'ai_model_latency_seconds',
            'Model inference latency',
            ['model_type', 'version']
        )

        # متریک‌های سیستمی
        self.gpu_memory = prom.Gauge(
            'gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['device']
        )
        self.cpu_usage = prom.Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )
        self.memory_usage = prom.Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes'
        )

    def _initialize_storage(self):
        """راه‌اندازی ساختارهای ذخیره‌سازی داده"""
        # ذخیره تاریخچه متریک‌ها
        self.metric_history = defaultdict(
            lambda: deque(maxlen=self.config.history_size)
        )

        # آمار تجمیعی درخواست‌ها
        self.request_stats = defaultdict(lambda: {
            'total': 0,
            'success': 0,
            'failure': 0,
            'durations': deque(maxlen=1000),
            'last_updated': datetime.now()
        })

        # وضعیت مدل‌ها
        self.model_stats = defaultdict(lambda: {
            'accuracy': 0.0,
            'latency': deque(maxlen=100),
            'error_rate': 0.0,
            'last_updated': datetime.now()
        })

    def record_request(self, model_type: str, duration: float, success: bool):
        """
        ثبت اطلاعات یک درخواست

        Args:
            model_type: نوع مدل
            duration: مدت زمان پردازش (ثانیه)
            success: موفقیت یا شکست درخواست
        """
        # بروزرسانی متریک‌های Prometheus
        status = 'success' if success else 'failure'
        self.request_counter.labels(model_type=model_type, status=status).inc()
        self.request_duration.labels(model_type=model_type).observe(duration)

        # بروزرسانی آمار داخلی
        stats = self.request_stats[model_type]
        stats['total'] += 1
        stats['success' if success else 'failure'] += 1
        stats['durations'].append(duration)
        stats['last_updated'] = datetime.now()

        self._update_model_metrics(model_type)

    def record_model_metrics(self, model_type: str, metrics: Dict[str, Any]):
        """
        ثبت متریک‌های مربوط به عملکرد مدل

        Args:
            model_type: نوع مدل
            metrics: دیکشنری متریک‌های مدل
        """
        # بروزرسانی متریک‌های Prometheus
        if 'accuracy' in metrics:
            self.model_accuracy.labels(
                model_type=model_type,
                version=metrics.get('version', 'unknown')
            ).set(metrics['accuracy'])

        if 'latency' in metrics:
            self.model_latency.labels(
                model_type=model_type,
                version=metrics.get('version', 'unknown')
            ).set(metrics['latency'])

        # بروزرسانی آمار داخلی
        stats = self.model_stats[model_type]
        stats['accuracy'] = metrics.get('accuracy', stats['accuracy'])
        if 'latency' in metrics:
            stats['latency'].append(metrics['latency'])
        stats['error_rate'] = metrics.get('error_rate', stats['error_rate'])
        stats['last_updated'] = datetime.now()

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های سیستمی"""
        metrics = {
            'cpu': {
                'usage_percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'load_avg': psutil.getloadavg()
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent
            },
            'timestamp': datetime.now().isoformat()
        }

        # اضافه کردن متریک‌های GPU در صورت وجود
        if torch.cuda.is_available():
            metrics['gpu'] = {}
            for i in range(torch.cuda.device_count()):
                metrics['gpu'][f'device_{i}'] = {
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_cached': torch.cuda.memory_cached(i),
                    'utilization': torch.cuda.utilization(i)
                }

        # بروزرسانی متریک‌های Prometheus
        self.cpu_usage.set(metrics['cpu']['usage_percent'])
        self.memory_usage.set(metrics['memory']['used'])

        if 'gpu' in metrics:
            for device, stats in metrics['gpu'].items():
                self.gpu_memory.labels(device=device).set(stats['memory_allocated'])

        return metrics

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های تجمیع شده"""
        metrics = {
            'system': self._get_system_metrics(),
            'models': self._get_model_metrics(),
            'requests': self._get_request_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        return metrics

    def _update_model_metrics(self, model_type: str):
        """بروزرسانی متریک‌های محاسباتی مدل"""
        stats = self.request_stats[model_type]
        if stats['total'] > 0:
            # محاسبه نرخ موفقیت
            success_rate = stats['success'] / stats['total']

            # محاسبه زمان پاسخ متوسط
            avg_duration = (
                sum(stats['durations']) / len(stats['durations'])
                if stats['durations'] else 0
            )

            # ذخیره در تاریخچه
            self.metric_history[f'{model_type}_success_rate'].append(
                (datetime.now(), success_rate)
            )
            self.metric_history[f'{model_type}_avg_duration'].append(
                (datetime.now(), avg_duration)
            )

    def _cleanup_old_metrics(self):
        """پاکسازی متریک‌های قدیمی"""
        cutoff_time = datetime.now() - timedelta(
            seconds=self.config.history_size * self.config.aggregation_interval
        )

        for metric_name, history in self.metric_history.items():
            while history and history[0][0] < cutoff_time:
                history.popleft()

    def get_metric_history(self, metric_name: str,
                           duration: Optional[timedelta] = None) -> List[tuple]:
        """
        دریافت تاریخچه یک متریک خاص

        Args:
            metric_name: نام متریک
            duration: مدت زمان مورد نظر (اختیاری)

        Returns:
            لیست تاپل‌های (زمان، مقدار) برای متریک
        """
        if metric_name not in self.metric_history:
            return []

        history = self.metric_history[metric_name]
        if not duration:
            return list(history)

        cutoff_time = datetime.now() - duration
        return [(t, v) for t, v in history if t >= cutoff_time]