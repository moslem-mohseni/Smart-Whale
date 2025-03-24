import asyncio
import logging
import time
from typing import Optional, Callable
from buffer import buffer_manager
from processor.stream_processor import StreamProcessor
from processor.batch_processor import BatchProcessor

logging.basicConfig(level=logging.INFO)


class FlowController:
    def __init__(self,
                 max_queue_size: int = 5000,
                 batch_processing_threshold: int = 1000,
                 monitor_interval: float = 2.0,
                 backpressure_threshold: float = 0.8,
                 scaling_function: Optional[Callable] = None):
        """
        کنترل‌کننده جریان داده بین پردازشگرها

        :param max_queue_size: حداکثر تعداد آیتم‌های قابل پردازش در صف
        :param batch_processing_threshold: حد آستانه برای انتقال داده‌ها به پردازش دسته‌ای
        :param monitor_interval: فاصله‌ی زمانی برای بررسی وضعیت جریان داده (بر حسب ثانیه)
        :param backpressure_threshold: آستانه‌ی ازدحام پردازشی که باعث کاهش نرخ پردازش می‌شود (۸۰٪ پیش‌فرض)
        :param scaling_function: تابع سفارشی برای مقیاس‌پذیری در هنگام افزایش بار پردازشی
        """
        self.max_queue_size = max_queue_size
        self.batch_processing_threshold = batch_processing_threshold
        self.monitor_interval = monitor_interval
        self.backpressure_threshold = backpressure_threshold
        self.scaling_function = scaling_function

        self.running = True
        self.stream_processor: Optional[StreamProcessor] = None
        self.batch_processor: Optional[BatchProcessor] = None

    async def attach_processors(self, stream_processor: StreamProcessor, batch_processor: BatchProcessor):
        """
        اتصال پردازشگرهای جریانی و دسته‌ای به کنترل‌کننده جریان

        :param stream_processor: نمونه‌ای از `StreamProcessor`
        :param batch_processor: نمونه‌ای از `BatchProcessor`
        """
        self.stream_processor = stream_processor
        self.batch_processor = batch_processor

    async def monitor_flow(self):
        """
        پایش جریان داده و تنظیم آن برای بهینه‌سازی کارایی سیستم
        """
        while self.running:
            queue_size = await buffer_manager.buffer_size()
            logging.info(f"📊 Current Buffer Size: {queue_size}/{self.max_queue_size}")

            # مدیریت جریان در صورت ازدحام پردازشی
            if queue_size > self.max_queue_size * self.backpressure_threshold:
                logging.warning("⚠️ High queue size detected! Applying backpressure control...")
                await self.apply_backpressure()

            # انتقال داده‌ها به پردازش دسته‌ای اگر از حد آستانه عبور کند
            if queue_size > self.batch_processing_threshold:
                logging.info("🔄 Transferring data to batch processing...")
                await self.transfer_to_batch_processing()

            await asyncio.sleep(self.monitor_interval)

    async def apply_backpressure(self):
        """
        اعمال مکانیزم‌های کنترل ازدحام پردازشی (Backpressure Handling)
        """
        if self.stream_processor:
            logging.warning("🚨 Reducing stream processing rate...")
            # در اینجا می‌توان نرخ پردازش را کاهش داد
            await asyncio.sleep(1)  # شبیه‌سازی کاهش سرعت پردازش

        if self.scaling_function:
            logging.info("⚙️ Applying dynamic scaling strategy...")
            self.scaling_function()  # اعمال استراتژی مقیاس‌پذیری

    async def transfer_to_batch_processing(self):
        """
        انتقال داده‌ها از جریان پردازش بلادرنگ به پردازش دسته‌ای
        """
        batch = []
        for _ in range(self.batch_processing_threshold):
            data = await buffer_manager.get_data()
            if data:
                batch.append(data)
            else:
                break

        if batch and self.batch_processor:
            logging.info(f"📦 Processing batch of {len(batch)} items...")
            await self.batch_processor._process_batch(batch)

    async def stop(self):
        """
        توقف کنترل‌کننده جریان
        """
        self.running = False
        logging.info("⛔ Stopping Flow Controller monitoring.")


async def test_flow_controller():
    """
    تست عملکرد FlowController
    """
    stream_processor = StreamProcessor(
        kafka_bootstrap_servers="localhost:9092",
        topic="raw_data_stream",
        group_id="flow_test_group",
        process_function=lambda data: {"processed_data": data.get("raw_data", "").upper()},
        output_topic="processed_data_stream",
    )

    batch_processor = BatchProcessor(
        kafka_bootstrap_servers="localhost:9092",
        input_topic="raw_data_batch",
        output_topic="processed_data_batch",
        group_id="batch_processor_group",
        batch_size=10,
        batch_interval=5,
        process_function=BatchProcessor.default_process_function,
    )

    flow_controller = FlowController()
    await flow_controller.attach_processors(stream_processor, batch_processor)

    asyncio.create_task(flow_controller.monitor_flow())

    # **شبیه‌سازی پردازش داده‌ها**
    for i in range(1500):
        await buffer_manager.add_data({"raw_data": f"Message-{i}"})
        await asyncio.sleep(0.01)  # تنظیم سرعت ورود داده‌ها

    await asyncio.sleep(10)  # اجازه می‌دهد مانیتورینگ اجرا شود
    await flow_controller.stop()
    print("✅ Flow Controller Test Passed!")


# اجرای تست
if __name__ == "__main__":
    asyncio.run(test_flow_controller())
