# ai/core/orchestration/load_balancer.py
"""
سیستم مدیریت و توزیع بار برای مدل‌های هوش مصنوعی

این ماژول مسئولیت مدیریت و توزیع هوشمند درخواست‌ها بین نمونه‌های مختلف مدل‌ها را بر عهده دارد.
همچنین وظیفه مدیریت منابع و تصمیم‌گیری برای مقیاس‌پذیری را انجام می‌دهد.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelInstance:
    """نگهدارنده اطلاعات یک نمونه از مدل"""
    instance_id: str
    model_type: str
    created_at: datetime
    status: str = 'initializing'
    current_load: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_error: Optional[str] = None
    gpu_memory_used: Optional[float] = None
    cpu_usage: Optional[float] = None


class LoadBalancer:
    """مدیریت توزیع بار و مقیاس‌پذیری مدل‌ها"""

    def __init__(self):
        self.instances: Dict[str, ModelInstance] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self._monitor_task: Optional[asyncio.Task] = None
        self._should_stop = False

        # تنظیمات مقیاس‌پذیری
        self.min_instances = 1
        self.max_instances = 10
        self.scale_up_threshold = 0.8  # 80% بار
        self.scale_down_threshold = 0.3  # 30% بار
        self.cooldown_period = 300  # 5 دقیقه

    async def start(self):
        """شروع سرویس توزیع بار"""
        if not self._monitor_task:
            self._should_stop = False
            self._monitor_task = asyncio.create_task(self._monitor_resources())
            logger.info("Load balancer started")

    async def stop(self):
        """توقف سرویس"""
        if self._monitor_task:
            self._should_stop = True
            await self._monitor_task
            self._monitor_task = None
            logger.info("Load balancer stopped")

    async def add_instance(self, model_type: str) -> str:
        """اضافه کردن یک نمونه جدید از مدل"""
        instance_id = f"{model_type}_{len(self.instances) + 1}"
        instance = ModelInstance(
            instance_id=instance_id,
            model_type=model_type,
            created_at=datetime.now()
        )
        self.instances[instance_id] = instance
        logger.info(f"Added new instance: {instance_id}")
        return instance_id

    async def remove_instance(self, instance_id: str):
        """حذف یک نمونه از مدل"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            logger.info(f"Removed instance: {instance_id}")

    async def get_instance(self, model_type: str) -> Optional[str]:
        """انتخاب بهترین نمونه برای پردازش درخواست جدید"""
        available_instances = [
            inst for inst in self.instances.values()
            if inst.model_type == model_type and inst.status == 'ready'
        ]

        if not available_instances:
            return None

        # انتخاب نمونه با کمترین بار
        selected = min(available_instances, key=lambda x: x.current_load)
        return selected.instance_id

    async def update_metrics(self, instance_id: str, metrics: Dict[str, Any]):
        """بروزرسانی متریک‌های یک نمونه"""
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            instance.current_load = metrics.get('current_load', instance.current_load)
            instance.gpu_memory_used = metrics.get('gpu_memory', instance.gpu_memory_used)
            instance.cpu_usage = metrics.get('cpu_usage', instance.cpu_usage)
            instance.average_response_time = metrics.get('avg_response_time',
                                                         instance.average_response_time)

    async def _monitor_resources(self):
        """نظارت مستمر بر منابع و تصمیم‌گیری برای مقیاس‌پذیری"""
        last_scale_time = datetime.now()

        while not self._should_stop:
            try:
                current_time = datetime.now()
                if (current_time - last_scale_time).total_seconds() > self.cooldown_period:
                    await self._check_scaling_needs()
                    last_scale_time = current_time

                # بررسی سلامت نمونه‌ها
                await self._check_instances_health()

                await asyncio.sleep(60)  # بررسی هر دقیقه

            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                await asyncio.sleep(60)

    async def _check_scaling_needs(self):
        """بررسی نیاز به تغییر مقیاس"""
        for model_type in set(inst.model_type for inst in self.instances.values()):
            instances = [inst for inst in self.instances.values()
                         if inst.model_type == model_type]

            avg_load = np.mean([inst.current_load for inst in instances])

            if avg_load > self.scale_up_threshold:
                # نیاز به افزایش ظرفیت
                if len(instances) < self.max_instances:
                    await self.add_instance(model_type)
                    logger.info(f"Scaling up {model_type} due to high load")

            elif avg_load < self.scale_down_threshold:
                # امکان کاهش ظرفیت
                if len(instances) > self.min_instances:
                    # حذف نمونه با کمترین بار
                    instance_to_remove = min(instances, key=lambda x: x.current_load)
                    await self.remove_instance(instance_to_remove.instance_id)
                    logger.info(f"Scaling down {model_type} due to low load")

    async def _check_instances_health(self):
        """بررسی سلامت نمونه‌ها"""
        for instance_id, instance in list(self.instances.items()):
            # بررسی وضعیت پاسخ‌دهی
            if instance.failed_requests / max(instance.total_requests, 1) > 0.2:
                # نرخ خطای بالا
                logger.warning(f"High error rate for instance {instance_id}")
                instance.status = 'error'

            # بررسی زمان پاسخ
            if instance.average_response_time > 2.0:  # بیشتر از 2 ثانیه
                logger.warning(f"High latency for instance {instance_id}")

    def get_status(self) -> Dict[str, Any]:
        """دریافت وضعیت کلی سیستم"""
        return {
            'total_instances': len(self.instances),
            'instances_by_type': {
                model_type: len([i for i in self.instances.values()
                                 if i.model_type == model_type])
                for model_type in set(i.model_type for i in self.instances.values())
            },
            'total_requests': sum(i.total_requests for i in self.instances.values()),
            'average_response_time': np.mean([i.average_response_time
                                              for i in self.instances.values()
                                              if i.average_response_time > 0]),
            'error_rate': sum(i.failed_requests for i in self.instances.values()) /
                          max(sum(i.total_requests for i in self.instances.values()), 1)
        }