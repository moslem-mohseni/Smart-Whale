from prometheus_client import Counter, Gauge, Histogram


class FileMetrics:
    """
    مانیتورینگ و جمع‌آوری متریک‌های مربوط به مدیریت فایل
    """

    def __init__(self):
        self.upload_counter = Counter("file_upload_count", "Total number of uploaded files")
        self.download_counter = Counter("file_download_count", "Total number of downloaded files")
        self.delete_counter = Counter("file_delete_count", "Total number of deleted files")
        self.file_size_gauge = Gauge("file_size_bytes", "Size of last uploaded file in bytes")
        self.upload_latency_histogram = Histogram("file_upload_latency_seconds", "Histogram of file upload latency")

    def record_upload(self, file_size: int, latency: float):
        """ثبت اطلاعات مربوط به آپلود فایل"""
        self.upload_counter.inc()
        self.file_size_gauge.set(file_size)
        self.upload_latency_histogram.observe(latency)

    def record_download(self):
        """ثبت اطلاعات مربوط به دانلود فایل"""
        self.download_counter.inc()

    def record_delete(self):
        """ثبت اطلاعات مربوط به حذف فایل"""
        self.delete_counter.inc()