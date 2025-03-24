"""
ثابت‌ها و مقادیر پیش‌فرض برای سیستم پیام‌رسانی
"""

# نسخه ماژول پیام‌رسانی
MESSAGING_VERSION = "1.0.0"

# پیشوندهای موضوعات کافکا
TOPIC_PREFIX = "smartwhale"
DATA_PREFIX = f"{TOPIC_PREFIX}.data"
MODELS_PREFIX = f"{TOPIC_PREFIX}.models"
BALANCE_PREFIX = f"{TOPIC_PREFIX}.balance"
CORE_PREFIX = f"{TOPIC_PREFIX}.core"

# نام‌های موضوعات استاندارد
DATA_REQUESTS_TOPIC = f"{DATA_PREFIX}.requests"
DATA_RESPONSES_PREFIX = f"{DATA_PREFIX}.responses"
MODELS_REQUESTS_TOPIC = f"{MODELS_PREFIX}.requests"
BALANCE_METRICS_TOPIC = f"{BALANCE_PREFIX}.metrics"
BALANCE_EVENTS_TOPIC = f"{BALANCE_PREFIX}.events"
SYSTEM_LOGS_TOPIC = f"{CORE_PREFIX}.logs"

# تنظیمات پیش‌فرض برای موضوعات
DEFAULT_PARTITIONS = 5
DEFAULT_REPLICATION = 3
MODEL_TOPIC_PARTITIONS = 3
MODEL_TOPIC_REPLICATION = 2

# مقادیر پیش‌فرض برای اولویت‌ها
PRIORITY_CRITICAL = 1
PRIORITY_HIGH = 2
PRIORITY_MEDIUM = 3
PRIORITY_LOW = 4
PRIORITY_BACKGROUND = 5
DEFAULT_PRIORITY = PRIORITY_MEDIUM

# زمان انتظار پیش‌فرض برای عملیات‌ها (به ثانیه)
DEFAULT_OPERATION_TIMEOUT = 30
DEFAULT_KAFKA_TIMEOUT = 10

# اندازه‌های پیش‌فرض
DEFAULT_BATCH_SIZE = 100
DEFAULT_BUFFER_SIZE = 1024 * 1024  # 1MB

# منابع درخواست
REQUEST_SOURCE_USER = "user"        # درخواست از کاربر
REQUEST_SOURCE_MODEL = "model"      # درخواست داخلی از مدل
REQUEST_SOURCE_SYSTEM = "system"    # درخواست سیستمی
DEFAULT_REQUEST_SOURCE = REQUEST_SOURCE_USER

# وضعیت‌های پیام
MESSAGE_STATUS_SUCCESS = "success"
MESSAGE_STATUS_ERROR = "error"
MESSAGE_STATUS_PENDING = "pending"
MESSAGE_STATUS_PARTIAL = "partial"
MESSAGE_STATUS_TIMEOUT = "timeout"

# برچسب‌های خطا
ERROR_INVALID_REQUEST = "invalid_request"
ERROR_NO_DATA_FOUND = "no_data_found"
ERROR_SOURCE_UNAVAILABLE = "source_unavailable"
ERROR_PROCESSING_FAILED = "processing_failed"
ERROR_TIMEOUT = "timeout"
ERROR_UNKNOWN = "unknown_error"
