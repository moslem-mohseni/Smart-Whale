import asyncio
from typing import List, Dict, Any


class BatchScheduler:
    """
    این کلاس مسئول زمان‌بندی و مدیریت پردازش دسته‌های پردازشی است.
    """

    def __init__(self):
        """
        مقداردهی اولیه و ایجاد صف دسته‌های پردازشی.
        """
        self.batch_queue = asyncio.PriorityQueue()

    async def schedule_batch(self, batch_data: Dict[str, Any], priority: int = 1):
        """
        اضافه کردن دسته جدید به صف پردازشی با اولویت مشخص.
        """
        await self.batch_queue.put((priority, batch_data))

    async def process_batches(self, process_function):
        """
        پردازش دسته‌های موجود در صف پردازشی بر اساس اولویت.
        """
        while True:
            priority, batch_data = await self.batch_queue.get()
            print(f"🔄 پردازش دسته با اولویت {priority} و اندازه {len(batch_data)}")
            await process_function(batch_data)
            self.batch_queue.task_done()
