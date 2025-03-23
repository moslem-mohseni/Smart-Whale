from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum
import statistics


@dataclass
class LearningMetric:
    """متریک‌های مربوط به یادگیری"""
    query_time: float  # زمان پردازش پرس‌وجو
    source_count: int  # تعداد منابع استفاده شده
    success_rate: float  # نرخ موفقیت
    token_count: int  # تعداد توکن‌های پردازش شده
    confidence_score: float  # نمره اطمینان
    response_length: int  # طول پاسخ
    timestamp: datetime  # زمان ثبت


@dataclass
class PerformanceMetric:
    """متریک‌های عملکردی"""
    cpu_usage: float  # مصرف CPU
    memory_usage: float  # مصرف حافظه
    gpu_usage: Optional[float]  # مصرف GPU
    latency: float  # تأخیر
    throughput: float  # توان عملیاتی
    timestamp: datetime  # زمان ثبت


@dataclass
class ResourceMetric:
    """متریک‌های منابع"""
    source_id: str  # شناسه منبع
    requests_count: int  # تعداد درخواست‌ها
    success_count: int  # تعداد موفقیت‌ها
    error_count: int  # تعداد خطاها
    average_response_time: float  # میانگین زمان پاسخ
    timestamp: datetime  # زمان ثبت


class MetricsCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learning_metrics: List[LearningMetric] = []
        self.performance_metrics: List[PerformanceMetric] = []
        self.resource_metrics: Dict[str, List[ResourceMetric]] = {}
        self._collection_task = None
        self._running = False

    async def start_collection(self) -> None:
        """شروع جمع‌آوری خودکار متریک‌ها"""
        self._running = True
        self._collection_task = asyncio.create_task(self._collect_metrics())

    async def stop_collection(self) -> None:
        """توقف جمع‌آوری متریک‌ها"""
        self._running = False
        if self._collection_task:
            await self._collection_task

    async def _collect_metrics(self) -> None:
        """جمع‌آوری دوره‌ای متریک‌ها"""
        interval = self.config.get('collection_interval', 60)
        while self._running:
            try:
                perf_metric = await self._collect_performance_metrics()
                self.performance_metrics.append(perf_metric)

                # نگهداری فقط متریک‌های یک روز اخیر
                cutoff = datetime.now() - timedelta(days=1)
                self.performance_metrics = [
                    m for m in self.performance_metrics
                    if m.timestamp > cutoff
                ]

                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                await asyncio.sleep(5)

    async def _collect_performance_metrics(self) -> PerformanceMetric:
        """جمع‌آوری متریک‌های عملکردی"""
        import psutil
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else None
        except:
            gpu_usage = None

        return PerformanceMetric(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            gpu_usage=gpu_usage,
            latency=await self._measure_latency(),
            throughput=await self._measure_throughput(),
            timestamp=datetime.now()
        )

    async def _measure_latency(self) -> float:
        """اندازه‌گیری تأخیر سیستم"""
        start_time = time.time()
        # اینجا یک عملیات نمونه انجام می‌دهیم
        await asyncio.sleep(0.1)
        return time.time() - start_time

    async def _measure_throughput(self) -> float:
        """اندازه‌گیری توان عملیاتی سیستم"""
        # محاسبه تعداد درخواست‌های موفق در واحد زمان
        return len([m for m in self.learning_metrics
                    if m.timestamp > datetime.now() - timedelta(minutes=1)])

    def record_learning_metric(self, metric: LearningMetric) -> None:
        """ثبت متریک یادگیری"""
        self.learning_metrics.append(metric)
        # نگهداری فقط متریک‌های یک روز اخیر
        cutoff = datetime.now() - timedelta(days=1)
        self.learning_metrics = [
            m for m in self.learning_metrics
            if m.timestamp > cutoff
        ]

    def record_resource_metric(self, metric: ResourceMetric) -> None:
        """ثبت متریک منبع"""
        if metric.source_id not in self.resource_metrics:
            self.resource_metrics[metric.source_id] = []

        self.resource_metrics[metric.source_id].append(metric)

        # نگهداری فقط متریک‌های یک روز اخیر
        cutoff = datetime.now() - timedelta(days=1)
        self.resource_metrics[metric.source_id] = [
            m for m in self.resource_metrics[metric.source_id]
            if m.timestamp > cutoff
        ]

    def get_learning_statistics(self) -> Dict[str, float]:
        """محاسبه آمار یادگیری"""
        if not self.learning_metrics:
            return {}

        query_times = [m.query_time for m in self.learning_metrics]
        success_rates = [m.success_rate for m in self.learning_metrics]
        confidence_scores = [m.confidence_score for m in self.learning_metrics]

        return {
            'avg_query_time': statistics.mean(query_times),
            'median_query_time': statistics.median(query_times),
            'avg_success_rate': statistics.mean(success_rates),
            'avg_confidence': statistics.mean(confidence_scores),
            'total_queries': len(self.learning_metrics)
        }

    def get_performance_statistics(self) -> Dict[str, float]:
        """محاسبه آمار عملکردی"""
        if not self.performance_metrics:
            return {}

        cpu_usage = [m.cpu_usage for m in self.performance_metrics]
        memory_usage = [m.memory_usage for m in self.performance_metrics]
        latencies = [m.latency for m in self.performance_metrics]

        return {
            'avg_cpu_usage': statistics.mean(cpu_usage),
            'max_cpu_usage': max(cpu_usage),
            'avg_memory_usage': statistics.mean(memory_usage),
            'max_memory_usage': max(memory_usage),
            'avg_latency': statistics.mean(latencies),
            'p95_latency': self._percentile(latencies, 95)
        }

    def get_resource_statistics(self, source_id: str) -> Dict[str, float]:
        """محاسبه آمار منبع"""
        if source_id not in self.resource_metrics:
            return {}

        metrics = self.resource_metrics[source_id]
        if not metrics:
            return {}

        response_times = [m.average_response_time for m in metrics]
        success_rates = [m.success_count / (m.requests_count or 1) for m in metrics]

        return {
            'total_requests': sum(m.requests_count for m in metrics),
            'total_errors': sum(m.error_count for m in metrics),
            'avg_response_time': statistics.mean(response_times),
            'avg_success_rate': statistics.mean(success_rates)
        }

    def _percentile(self, data: List[float], percentile: int) -> float:
        """محاسبه صدک مشخص"""
        size = len(data)
        sorted_data = sorted(data)
        index = (size * percentile) // 100
        return sorted_data[index]

    def export_metrics(self) -> Dict[str, Any]:
        """خروجی گرفتن از تمام متریک‌ها"""
        return {
            'learning_metrics': [asdict(m) for m in self.learning_metrics],
            'performance_metrics': [asdict(m) for m in self.performance_metrics],
            'resource_metrics': {
                source_id: [asdict(m) for m in metrics]
                for source_id, metrics in self.resource_metrics.items()
            }
        }

    def import_metrics(self, data: Dict[str, Any]) -> None:
        """وارد کردن متریک‌ها از فایل"""
        self.learning_metrics = [
            LearningMetric(**m) for m in data.get('learning_metrics', [])
        ]
        self.performance_metrics = [
            PerformanceMetric(**m) for m in data.get('performance_metrics', [])
        ]
        self.resource_metrics = {
            source_id: [ResourceMetric(**m) for m in metrics]
            for source_id, metrics in data.get('resource_metrics', {}).items()
        }