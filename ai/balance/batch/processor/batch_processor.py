import asyncio
from typing import List, Dict, Any
from ai.core.resource_management.allocator import ResourceAllocator
from ai.core.scheduler.task_scheduler import TaskScheduler
from ai.monitoring.metrics_interface import MetricsInterface
from ai.data.stream.stream_processor import StreamProcessor
from ai.balance.batch.optimizer.batch_optimizer import BatchOptimizer
from ai.balance.batch.processor.request_merger import RequestMerger
from ai.balance.batch.processor.batch_splitter import BatchSplitter


class BatchProcessor:
    """
    پردازشگر دسته‌ای برای مدیریت درخواست‌ها و بهینه‌سازی پردازش‌های گروهی
    """

    def __init__(self):
        self.resource_allocator = ResourceAllocator()
        self.task_scheduler = TaskScheduler()
        self.metrics_interface = MetricsInterface()
        self.stream_processor = StreamProcessor()
        self.batch_optimizer = BatchOptimizer()
        self.request_merger = RequestMerger()
        self.batch_splitter = BatchSplitter()

    async def process_batch(self, batch_data: List[Dict[str, Any]], priority: int = 1) -> Dict[str, Any]:
        """
        پردازش داده‌های دسته‌ای، ترکیب درخواست‌های مشابه و تخصیص منابع
        """
        # 1️⃣ ادغام درخواست‌های مشابه برای بهینه‌سازی پردازش
        merged_data = self.request_merger.merge_requests(batch_data)

        # 2️⃣ تجزیه دسته‌های بسیار بزرگ برای پردازش مؤثرتر
        optimized_batches = self.batch_splitter.split_large_batches(merged_data)

        results = []

        for batch in optimized_batches:
            # 3️⃣ تخصیص منابع برای پردازش دسته‌ای
            required_resources = self.batch_optimizer.calculate_optimal_resources(batch)
            allocated_resources = self.resource_allocator.allocate_resources(
                cpu_demand=required_resources['cpu'],
                memory_demand=required_resources['memory']
            )

            if not allocated_resources:
                print("⚠️ منابع کافی برای پردازش این دسته موجود نیست! در حال انتظار...")
                await asyncio.sleep(1)
                continue

            # 4️⃣ پردازش دسته با استفاده از StreamProcessor
            batch_result = await self.stream_processor.process_stream(batch)
            results.append(batch_result)

            # 5️⃣ ثبت متریک‌های پردازشی در Monitoring
            self.metrics_interface.report_metrics({
                "batch_size": len(batch),
                "processing_time": batch_result.get("time_taken", 0),
                "resource_usage": allocated_resources
            })

        return {"status": "completed", "processed_batches": len(results), "details": results}

    async def handle_incoming_batches(self, batch_queue: asyncio.Queue):
        """
        مدیریت صف ورودی برای پردازش دسته‌ای و زمان‌بندی اجرای پردازش‌ها
        """
        while True:
            batch_data = await batch_queue.get()
            print(f"✅ دریافت دسته جدید با {len(batch_data)} درخواست")
            await self.task_scheduler.schedule_task(self.process_batch, priority=2, batch_data=batch_data)
            batch_queue.task_done()


# اجرای پردازشگر دسته‌ای در یک رویداد حلقه
if __name__ == "__main__":
    batch_queue = asyncio.Queue()
    processor = BatchProcessor()
    asyncio.run(processor.handle_incoming_batches(batch_queue))
