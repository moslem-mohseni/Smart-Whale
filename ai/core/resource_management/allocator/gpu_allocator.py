import torch

class GPUAllocator:
    def __init__(self, max_usage_percent=90):
        """
        مدیریت تخصیص پردازنده گرافیکی (GPU)
        :param max_usage_percent: حداکثر درصد مجاز استفاده از حافظه GPU (پیش‌فرض: 90٪)
        """
        self.max_usage_percent = max_usage_percent
        self.available_gpus = torch.cuda.device_count()

    def get_gpu_status(self):
        """
        دریافت وضعیت GPUهای موجود
        :return: لیستی از اطلاعات GPUهای در دسترس
        """
        if self.available_gpus == 0:
            return "❌ هیچ GPU در دسترس نیست."

        gpu_status = []
        for i in range(self.available_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = gpu_info.total_memory
            memory_used_percent = (memory_allocated / memory_total) * 100

            gpu_status.append({
                "gpu_id": i,
                "name": gpu_info.name,
                "total_memory": memory_total,
                "used_memory": memory_allocated,
                "reserved_memory": memory_reserved,
                "percent_usage": round(memory_used_percent, 2)
            })

        return gpu_status

    def get_best_available_gpu(self):
        """
        پیدا کردن GPU با کمترین میزان مصرف حافظه
        :return: شماره GPU با کمترین مصرف حافظه یا None در صورت نبود GPU
        """
        if self.available_gpus == 0:
            return None

        best_gpu = None
        lowest_usage = 100  # درصد مصرف حافظه
        for i in range(self.available_gpus):
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            memory_used_percent = (memory_allocated / memory_total) * 100

            if memory_used_percent < self.max_usage_percent and memory_used_percent < lowest_usage:
                lowest_usage = memory_used_percent
                best_gpu = i

        return best_gpu

    def allocate_gpu(self):
        """
        تخصیص GPU با کمترین مصرف حافظه
        :return: شماره GPU تخصیص‌یافته یا پیام خطا
        """
        best_gpu = self.get_best_available_gpu()
        if best_gpu is not None:
            return f"✅ GPU-{best_gpu} با کمترین مصرف حافظه تخصیص یافت."
        else:
            return "❌ هیچ GPU مناسبی برای تخصیص در دسترس نیست!"
