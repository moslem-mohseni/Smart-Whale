import time
from core.cache.manager import CacheManager
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resource_management.allocator.cpu_allocator import CPUAllocator
from core.resource_management.allocator.memory_allocator import MemoryAllocator
from core.resource_management.allocator.gpu_allocator import GPUAllocator
from core.resource_management.monitor.threshold_manager import ThresholdManager
from core.monitoring.metrics.collector import MetricsCollector
from core.monitoring.metrics.exporter import MetricsExporter


class ResourcePredictor:
    """
    پیش‌بینی نیاز منابع برای مدل‌ها شامل CPU، RAM و GPU.
    """

    def __init__(self):
        self.cache = CacheManager()
        self.resource_monitor = ResourceMonitor()
        self.cpu_allocator = CPUAllocator()
        self.memory_allocator = MemoryAllocator()
        self.gpu_allocator = GPUAllocator()
        self.threshold_manager = ThresholdManager()
        self.metrics_collector = MetricsCollector()
        self.metrics_exporter = MetricsExporter()

    async def predict_resources(self, model_id: str, recent_usage: list) -> dict:
        """
        پیش‌بینی میزان منابع موردنیاز مدل مشخص‌شده.

        :param model_id: شناسه مدل مورد بررسی
        :param recent_usage: لیست مصرف‌های اخیر مدل
        :return: دیکشنری شامل پیش‌بینی مقدار مصرف منابع (CPU، RAM، GPU)
        """
        cache_key = f"resource_needs:{model_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        system_load = await self.resource_monitor.get_system_load()
        cpu_needs = await self.cpu_allocator.estimate_demand(model_id, recent_usage, system_load)
        memory_needs = await self.memory_allocator.estimate_demand(model_id, recent_usage, system_load)
        gpu_needs = await self.gpu_allocator.estimate_demand(model_id, recent_usage, system_load)

        threshold_data = await self.threshold_manager.get_thresholds(model_id)

        predicted_resources = self._calculate_final_resources(cpu_needs, memory_needs, gpu_needs, threshold_data)

        self.cache.set(cache_key, predicted_resources, ttl=600)
        self.metrics_collector.record("resource_needs_prediction", predicted_resources)
        self.metrics_exporter.export(predicted_resources)

        return predicted_resources

    def _calculate_final_resources(self, cpu_needs, memory_needs, gpu_needs, threshold_data) -> dict:
        """
        نهایی‌سازی مقدار منابع موردنیاز با در نظر گرفتن محدودیت‌ها.

        :param cpu_needs: پیش‌بینی نیاز CPU
        :param memory_needs: پیش‌بینی نیاز RAM
        :param gpu_needs: پیش‌بینی نیاز GPU
        :param threshold_data: اطلاعات آستانه‌ها
        :return: دیکشنری شامل مقدار نهایی منابع موردنیاز
        """
        final_cpu = min(cpu_needs, threshold_data["max_cpu"])
        final_memory = min(memory_needs, threshold_data["max_memory"])
        final_gpu = min(gpu_needs, threshold_data["max_gpu"])

        return {
            "predicted_cpu": final_cpu,
            "predicted_memory": final_memory,
            "predicted_gpu": final_gpu,
            "timestamp": time.time()
        }
