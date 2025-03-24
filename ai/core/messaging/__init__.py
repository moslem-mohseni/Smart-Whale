"""
ماژول مرکزی پیام‌رسانی برای ارتباط بین ماژول‌های سیستم
"""

# نسخه ماژول
__version__ = "1.0.0"

# صادر کردن توابع و کلاس‌های اصلی
from .constants import (
    # نسخه
    MESSAGING_VERSION,

    # موضوعات کافکا
    TOPIC_PREFIX, DATA_PREFIX, MODELS_PREFIX, BALANCE_PREFIX, CORE_PREFIX,
    DATA_REQUESTS_TOPIC, DATA_RESPONSES_PREFIX, MODELS_REQUESTS_TOPIC,
    BALANCE_METRICS_TOPIC, BALANCE_EVENTS_TOPIC, SYSTEM_LOGS_TOPIC,

    # تنظیمات پیش‌فرض موضوعات
    DEFAULT_PARTITIONS, DEFAULT_REPLICATION,
    MODEL_TOPIC_PARTITIONS, MODEL_TOPIC_REPLICATION,

    # اولویت‌های پیش‌فرض
    PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW, PRIORITY_BACKGROUND,
    DEFAULT_PRIORITY,

    # زمان‌های پیش‌فرض
    DEFAULT_OPERATION_TIMEOUT, DEFAULT_KAFKA_TIMEOUT,

    # اندازه‌های پیش‌فرض
    DEFAULT_BATCH_SIZE, DEFAULT_BUFFER_SIZE,

    # منابع درخواست
    REQUEST_SOURCE_USER, REQUEST_SOURCE_MODEL, REQUEST_SOURCE_SYSTEM,
    DEFAULT_REQUEST_SOURCE,

    # وضعیت‌های پیام
    MESSAGE_STATUS_SUCCESS, MESSAGE_STATUS_ERROR, MESSAGE_STATUS_PENDING,
    MESSAGE_STATUS_PARTIAL, MESSAGE_STATUS_TIMEOUT,

    # برچسب‌های خطا
    ERROR_INVALID_REQUEST, ERROR_NO_DATA_FOUND, ERROR_SOURCE_UNAVAILABLE,
    ERROR_PROCESSING_FAILED, ERROR_TIMEOUT, ERROR_UNKNOWN
)

from .message_schemas import (
    # انواع داده
    DataType, DataSource, OperationType, Priority, RequestSource,

    # کلاس‌های ساختار پیام
    MessageMetadata, RequestPayload, ResponsePayload,
    DataRequest, DataResponse,

    # توابع کمکی
    create_data_request, create_data_response,
    is_valid_data_request, is_valid_data_response
)

from .topic_manager import (
    TopicCategory, TopicManager
)

from .kafka_service import (
    KafkaService, kafka_service
)

# لیست نمادهای قابل صادر شدن
__all__ = [
    # نسخه
    "__version__", "MESSAGING_VERSION",

    # انواع داده و عملیات
    "DataType", "DataSource", "OperationType", "Priority", "RequestSource",

    # موضوعات
    "TopicCategory", "TopicManager",
    "TOPIC_PREFIX", "DATA_PREFIX", "MODELS_PREFIX", "BALANCE_PREFIX", "CORE_PREFIX",
    "DATA_REQUESTS_TOPIC", "DATA_RESPONSES_PREFIX", "MODELS_REQUESTS_TOPIC",
    "BALANCE_METRICS_TOPIC", "BALANCE_EVENTS_TOPIC", "SYSTEM_LOGS_TOPIC",

    # پیام‌ها
    "MessageMetadata", "RequestPayload", "ResponsePayload",
    "DataRequest", "DataResponse",
    "create_data_request", "create_data_response",
    "is_valid_data_request", "is_valid_data_response",

    # اولویت‌ها
    "PRIORITY_CRITICAL", "PRIORITY_HIGH", "PRIORITY_MEDIUM", "PRIORITY_LOW", "PRIORITY_BACKGROUND",
    "DEFAULT_PRIORITY",

    # سرویس کافکا
    "KafkaService", "kafka_service",

    # منابع درخواست
    "REQUEST_SOURCE_USER", "REQUEST_SOURCE_MODEL", "REQUEST_SOURCE_SYSTEM",

    # وضعیت‌های پیام
    "MESSAGE_STATUS_SUCCESS", "MESSAGE_STATUS_ERROR", "MESSAGE_STATUS_PENDING",
    "MESSAGE_STATUS_PARTIAL", "MESSAGE_STATUS_TIMEOUT"
]
