import numpy as np
from core.monitoring.metrics.collector import MetricsCollector
from core.resource_management.monitor.resource_monitor import ResourceMonitor

class BottleneckDetector:
    """
    ماژولی برای شناسایی گلوگاه‌های پردازشی در سیستم.
    این ماژول متریک‌های پردازشی را بررسی کرده و منابعی را که باعث کاهش عملکرد شده‌اند، شناسایی می‌کند.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
        self.bottleneck_thresholds = {
            "cpu_usage": 0.85,       # اگر مصرف CPU بیش از 85% باشد، ممکن است گلوگاه باشد
            "memory_usage": 0.90,    # اگر مصرف حافظه بیش از 90% باشد، ممکن است گلوگاه باشد
            "disk_io": 0.80,         # اگر استفاده از دیسک بالای 80% باشد، ممکن است گلوگاه باشد
            "network_latency": 200    # اگر تأخیر شبکه بیش از 200 میلی‌ثانیه باشد، مشکل ایجاد می‌شود
        }

    async def detect_bottlenecks(self) -> dict:
        """
        شناسایی گلوگاه‌های پردازشی سیستم.

        :return: دیکشنری شامل پردازش‌هایی که باعث کاهش سرعت شده‌اند.
        """
        # جمع‌آوری متریک‌های مختلف از سیستم
        cpu_usage = await self.resource_monitor.get_cpu_usage()
        memory_usage = await self.resource_monitor.get_memory_usage()
        disk_io = await self.metrics_collector.get_metric("disk_io")
        network_latency = await self.metrics_collector.get_metric("network_latency")

        # شناسایی مشکلات پردازشی
        bottlenecks = self._analyze_bottlenecks(cpu_usage, memory_usage, disk_io, network_latency)

        return bottlenecks

    def _analyze_bottlenecks(self, cpu_usage: list, memory_usage: list, disk_io: list, network_latency: list) -> dict:
        """
        تحلیل داده‌های منابع و شناسایی گلوگاه‌های احتمالی.

        :param cpu_usage: میزان مصرف CPU
        :param memory_usage: میزان مصرف حافظه
        :param disk_io: میزان استفاده از I/O دیسک
        :param network_latency: تأخیر در شبکه
        :return: دیکشنری از منابعی که باعث کاهش عملکرد شده‌اند.
        """
        bottlenecks = {}

        # بررسی مصرف CPU
        high_cpu_usage = [
            usage for usage in cpu_usage if usage > self.bottleneck_thresholds["cpu_usage"]
        ]
        if high_cpu_usage:
            bottlenecks["cpu"] = {
                "max_usage": max(high_cpu_usage),
                "affected_processes": high_cpu_usage
            }

        # بررسی مصرف حافظه
        high_memory_usage = [
            usage for usage in memory_usage if usage > self.bottleneck_thresholds["memory_usage"]
        ]
        if high_memory_usage:
            bottlenecks["memory"] = {
                "max_usage": max(high_memory_usage),
                "affected_processes": high_memory_usage
            }

        # بررسی I/O دیسک
        high_disk_io = [
            io for io in disk_io if io > self.bottleneck_thresholds["disk_io"]
        ]
        if high_disk_io:
            bottlenecks["disk_io"] = {
                "max_io": max(high_disk_io),
                "affected_processes": high_disk_io
            }

        # بررسی تأخیر شبکه
        high_network_latency = [
            latency for latency in network_latency if latency > self.bottleneck_thresholds["network_latency"]
        ]
        if high_network_latency:
            bottlenecks["network"] = {
                "max_latency": max(high_network_latency),
                "affected_processes": high_network_latency
            }

        return bottlenecks
