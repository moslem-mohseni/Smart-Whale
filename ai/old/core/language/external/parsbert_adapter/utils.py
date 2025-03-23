import torch
import gc
import logging

logger = logging.getLogger(__name__)


def manage_memory():
    """مدیریت حافظه GPU و پاکسازی کش در صورت نیاز"""
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024 ** 3  # تبدیل به گیگابایت
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

        if current_memory > 0.75 * total_memory:
            logger.warning("GPU memory usage exceeded 75%, clearing cache...")
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU cache cleared successfully.")


def check_gpu_status():
    """بررسی وضعیت GPU و میزان حافظه باقی‌مانده"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.info(
            f"GPU Memory Usage: {memory_allocated:.2f}GB / {memory_total:.2f}GB (Reserved: {memory_reserved:.2f}GB)")
    else:
        logger.info("CUDA is not available, using CPU.")


def estimate_available_memory():
    """برآورد میزان حافظه قابل استفاده GPU"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
        available_memory = total_memory - allocated_memory
        return available_memory
    return None
