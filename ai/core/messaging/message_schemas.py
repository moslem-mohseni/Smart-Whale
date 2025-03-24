"""
ساختارهای پیام و قراردادهای ارتباطی برای سیستم پیام‌رسانی
"""
import uuid
import json
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime

from ai.core.messaging.constants import (
    DEFAULT_PRIORITY, DEFAULT_REQUEST_SOURCE, DEFAULT_OPERATION_TIMEOUT,
    MESSAGE_STATUS_SUCCESS, MESSAGE_STATUS_ERROR
)


class DataType(Enum):
    """انواع داده‌های قابل جمع‌آوری"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    STRUCTURED = "structured"  # داده‌های ساختاریافته مانند JSON/CSV
    MIXED = "mixed"  # ترکیبی از انواع داده


class DataSource(Enum):
    """منابع داده قابل جمع‌آوری"""
    WEB = "web"  # صفحات وب عمومی
    WIKI = "wiki"  # ویکی‌پدیا و سایر ویکی‌ها
    TWITTER = "twitter"  # توییتر
    TELEGRAM = "telegram"  # تلگرام
    YOUTUBE = "youtube"  # یوتیوب
    APARAT = "aparat"  # آپارات
    NEWS = "news"  # خبرگزاری‌ها
    BOOK = "book"  # کتاب‌ها و منابع متنی
    API = "api"  # API‌های خارجی
    LINKEDIN = "linkedin"  # لینکدین
    PODCAST = "podcast"  # پادکست‌ها
    RSS = "rss"  # فیدهای RSS
    SCIENTIFIC = "scientific"  # منابع علمی
    DATABASE = "database"  # پایگاه‌های داده
    CUSTOM = "custom"  # منابع سفارشی‌سازی شده


class OperationType(Enum):
    """انواع عملیات قابل درخواست"""
    FETCH_DATA = "fetch_data"  # جمع‌آوری داده جدید
    PROCESS_DATA = "process_data"  # پردازش داده موجود
    UPDATE_DATA = "update_data"  # بروزرسانی داده‌های قبلی
    DELETE_DATA = "delete_data"  # حذف داده‌ها
    MONITOR_DATA = "monitor_data"  # پایش مداوم منبع داده
    VERIFY_DATA = "verify_data"  # بررسی صحت داده‌ها
    SEARCH_DATA = "search_data"  # جستجو در داده‌های موجود
    TRANSLATE_DATA = "translate_data"  # ترجمه داده‌ها
    SUMMARIZE_DATA = "summarize_data"  # خلاصه‌سازی داده‌ها
    ANALYZE_DATA = "analyze_data"  # تحلیل داده‌ها


class Priority(Enum):
    """سطوح اولویت پیام‌ها"""
    CRITICAL = 1  # بحرانی - اجرای فوری
    HIGH = 2  # بالا - اولویت بالاتر از متوسط
    MEDIUM = 3  # متوسط - اولویت پیش‌فرض
    LOW = 4  # پایین - می‌تواند به تأخیر بیفتد
    BACKGROUND = 5  # پس‌زمینه - فقط در زمان بی‌کاری سیستم


class RequestSource(Enum):
    """منبع درخواست‌ها"""
    USER = "user"  # درخواست از کاربر
    MODEL = "model"  # درخواست داخلی از مدل
    SYSTEM = "system"  # درخواست سیستمی
    SCHEDULED = "scheduled"  # درخواست زمان‌بندی شده
    API = "api"  # درخواست از API


@dataclass
class MessageMetadata:
    """ساختار متادیتای پایه برای همه پیام‌ها"""
    # شناسه‌های پیام
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # مبدأ و مقصد
    source: str = "balance"  # منبع پیام
    destination: str = "data"  # مقصد پیام

    # اولویت و مدیریت
    priority: int = DEFAULT_PRIORITY  # اولویت پیام (1 = بحرانی، 5 = پس‌زمینه)
    request_source: str = DEFAULT_REQUEST_SOURCE  # منبع درخواست (کاربر یا مدل)

    # اطلاعات تکمیلی
    correlation_id: Optional[str] = None  # برای ارتباط بین پیام‌های مرتبط
    trace_id: Optional[str] = None  # برای ردیابی دنباله پردازش‌ها
    session_id: Optional[str] = None  # شناسه نشست کاربر (در صورت وجود)

    # مدیریت خطا و بازپخش
    retry_count: int = 0  # تعداد تلاش‌های مجدد
    ttl: Optional[int] = None  # مدت اعتبار پیام (ثانیه)
    expires_at: Optional[str] = None  # زمان انقضای پیام

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل متادیتا به دیکشنری"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageMetadata':
        """ساخت نمونه MessageMetadata از دیکشنری"""
        # فیلترینگ کلید‌های معتبر
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class RequestPayload:
    """ساختار محتوای درخواست از Balance به Data"""
    # اطلاعات عملیات
    operation: Union[OperationType, str] = OperationType.FETCH_DATA
    model_id: str = ""  # شناسه مدل درخواست‌کننده

    # نوع و منبع داده
    data_type: Union[DataType, str] = DataType.TEXT  # نوع داده مورد نیاز
    data_source: Optional[Union[DataSource, str]] = None  # منبع داده (اختیاری)

    # پارامترهای درخواست
    parameters: Dict[str, Any] = field(default_factory=dict)  # پارامترهای اختصاصی
    response_topic: str = ""  # موضوع پاسخ

    # تنظیمات اضافی
    batch_id: Optional[str] = None  # شناسه دسته درخواست‌ها
    timeout: int = DEFAULT_OPERATION_TIMEOUT  # زمان انتظار برای پاسخ (ثانیه)
    max_size: Optional[int] = None  # حداکثر اندازه داده (بایت)
    require_fresh: bool = False  # آیا داده‌ها باید تازه باشند

    # فراداده
    tags: List[str] = field(default_factory=list)  # برچسب‌های کمکی
    context: Dict[str, Any] = field(default_factory=dict)  # زمینه درخواست

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل محتوا به دیکشنری"""
        result = asdict(self)
        # تبدیل مقادیر Enum به رشته
        if isinstance(self.operation, Enum):
            result["operation"] = self.operation.value
        if isinstance(self.data_type, Enum):
            result["data_type"] = self.data_type.value
        if isinstance(self.data_source, Enum) and self.data_source is not None:
            result["data_source"] = self.data_source.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestPayload':
        """ساخت نمونه RequestPayload از دیکشنری"""
        # پردازش مقادیر Enum
        if "operation" in data:
            try:
                data["operation"] = OperationType(data["operation"])
            except (ValueError, TypeError):
                pass

        if "data_type" in data:
            try:
                data["data_type"] = DataType(data["data_type"])
            except (ValueError, TypeError):
                pass

        if "data_source" in data and data["data_source"]:
            try:
                data["data_source"] = DataSource(data["data_source"])
            except (ValueError, TypeError):
                pass

        # فیلترینگ کلید‌های معتبر
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**filtered_data)


@dataclass
class ResponsePayload:
    """ساختار محتوای پاسخ از Data به Models"""
    # اطلاعات پاسخ
    operation: Union[OperationType, str] = OperationType.FETCH_DATA
    status: str = MESSAGE_STATUS_SUCCESS  # وضعیت پاسخ (success, error, partial)

    # مشخصات داده
    data_type: Union[DataType, str] = DataType.TEXT  # نوع داده
    data_source: Optional[Union[DataSource, str]] = None  # منبع داده

    # محتوای اصلی
    data: Any = None  # داده‌های اصلی
    error_message: Optional[str] = None  # پیام خطا (در صورت وجود)
    error_code: Optional[str] = None  # کد خطا (در صورت وجود)

    # اطلاعات تکمیلی
    metrics: Dict[str, Any] = field(default_factory=dict)  # متریک‌های پردازش
    batch_id: Optional[str] = None  # شناسه دسته درخواست‌ها
    next_cursor: Optional[str] = None  # برای پاسخ‌های صفحه‌بندی شده
    total_items: Optional[int] = None  # تعداد کل آیتم‌ها
    page_info: Dict[str, Any] = field(default_factory=dict)  # اطلاعات صفحه‌بندی

    # فراداده
    metadata: Dict[str, Any] = field(default_factory=dict)  # فراداده منبع
    source_url: Optional[str] = None  # آدرس منبع اصلی
    last_updated: Optional[str] = None  # آخرین بروزرسانی داده

    # اطلاعات پردازش
    processing_time: Optional[float] = None  # زمان پردازش (میلی‌ثانیه)
    cached: bool = False  # آیا از کش استفاده شده
    additional_info: Dict[str, Any] = field(default_factory=dict)  # اطلاعات تکمیلی

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل محتوا به دیکشنری"""
        result = asdict(self)
        # تبدیل مقادیر Enum به رشته
        if isinstance(self.operation, Enum):
            result["operation"] = self.operation.value
        if isinstance(self.data_type, Enum):
            result["data_type"] = self.data_type.value
        if isinstance(self.data_source, Enum) and self.data_source is not None:
            result["data_source"] = self.data_source.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponsePayload':
        """ساخت نمونه ResponsePayload از دیکشنری"""
        # پردازش مقادیر Enum
        if "operation" in data:
            try:
                data["operation"] = OperationType(data["operation"])
            except (ValueError, TypeError):
                pass

        if "data_type" in data:
            try:
                data["data_type"] = DataType(data["data_type"])
            except (ValueError, TypeError):
                pass

        if "data_source" in data and data["data_source"]:
            try:
                data["data_source"] = DataSource(data["data_source"])
            except (ValueError, TypeError):
                pass

        # فیلترینگ کلید‌های معتبر
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**filtered_data)


@dataclass
class DataRequest:
    """کلاس اصلی پیام درخواست از Balance به Data"""
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    payload: RequestPayload = field(default_factory=RequestPayload)

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل کل پیام به دیکشنری"""
        return {
            "metadata": self.metadata.to_dict(),
            "payload": self.payload.to_dict()
        }

    def to_json(self) -> str:
        """تبدیل پیام به JSON"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataRequest':
        """ساخت نمونه DataRequest از دیکشنری"""
        metadata = MessageMetadata.from_dict(data.get("metadata", {}))
        payload = RequestPayload.from_dict(data.get("payload", {}))
        return cls(metadata=metadata, payload=payload)

    @classmethod
    def from_json(cls, json_str: str) -> 'DataRequest':
        """ساخت نمونه DataRequest از JSON"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class DataResponse:
    """کلاس اصلی پیام پاسخ از Data به Models"""
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    payload: ResponsePayload = field(default_factory=ResponsePayload)

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل کل پیام به دیکشنری"""
        return {
            "metadata": self.metadata.to_dict(),
            "payload": self.payload.to_dict()
        }

    def to_json(self) -> str:
        """تبدیل پیام به JSON"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataResponse':
        """ساخت نمونه DataResponse از دیکشنری"""
        metadata = MessageMetadata.from_dict(data.get("metadata", {}))
        payload = ResponsePayload.from_dict(data.get("payload", {}))
        return cls(metadata=metadata, payload=payload)

    @classmethod
    def from_json(cls, json_str: str) -> 'DataResponse':
        """ساخت نمونه DataResponse از JSON"""
        data = json.loads(json_str)
        return cls.from_dict(data)


# توابع کمکی برای ساخت پیام‌ها

def create_data_request(
        model_id: str,
        data_type: Union[DataType, str] = DataType.TEXT,
        data_source: Optional[Union[DataSource, str]] = None,
        parameters: Dict[str, Any] = None,
        priority: Union[Priority, int] = Priority.MEDIUM,
        response_topic: str = "",
        operation: Union[OperationType, str] = OperationType.FETCH_DATA,
        request_source: Union[RequestSource, str] = RequestSource.USER
) -> DataRequest:
    """
    ایجاد یک درخواست داده برای ارسال به ماژول Data

    :param model_id: شناسه مدل درخواست‌کننده
    :param data_type: نوع داده مورد نیاز
    :param data_source: منبع داده (اختیاری)
    :param parameters: پارامترهای درخواست
    :param priority: اولویت درخواست
    :param response_topic: موضوع کافکا برای ارسال پاسخ
    :param operation: نوع عملیات
    :param request_source: منبع درخواست (کاربر یا مدل)
    :return: یک نمونه DataRequest
    """
    # استخراج مقدار عددی اولویت
    if isinstance(priority, Priority):
        priority_value = priority.value
    else:
        priority_value = int(priority)

    # استخراج مقدار رشته‌ای منبع درخواست
    if isinstance(request_source, RequestSource):
        source_value = request_source.value
    else:
        source_value = str(request_source)

    metadata = MessageMetadata(
        source="balance",
        destination="data",
        priority=priority_value,
        request_source=source_value
    )

    payload = RequestPayload(
        operation=operation,
        model_id=model_id,
        data_type=data_type,
        data_source=data_source,
        parameters=parameters or {},
        response_topic=response_topic
    )

    return DataRequest(metadata=metadata, payload=payload)


def create_data_response(
        request_id: str,
        model_id: str,
        status: str = MESSAGE_STATUS_SUCCESS,
        data: Any = None,
        data_type: Union[DataType, str] = DataType.TEXT,
        data_source: Optional[Union[DataSource, str]] = None,
        error_message: Optional[str] = None,
        metrics: Dict[str, Any] = None
) -> DataResponse:
    """
    ایجاد یک پاسخ داده برای ارسال به مدل‌ها

    :param request_id: شناسه درخواست اصلی
    :param model_id: شناسه مدل درخواست‌کننده
    :param status: وضعیت پاسخ (success, error, partial)
    :param data: داده‌های اصلی
    :param data_type: نوع داده
    :param data_source: منبع داده (اختیاری)
    :param error_message: پیام خطا (در صورت وجود)
    :param metrics: متریک‌های پردازش
    :return: یک نمونه DataResponse
    """
    metadata = MessageMetadata(
        request_id=request_id,
        source="data",
        destination=model_id
    )

    payload = ResponsePayload(
        status=status,
        data_type=data_type,
        data_source=data_source,
        data=data,
        error_message=error_message,
        metrics=metrics or {}
    )

    return DataResponse(metadata=metadata, payload=payload)


# توابع کمکی برای کار با پیام‌ها

def is_valid_data_request(request: Union[Dict[str, Any], DataRequest]) -> bool:
    """
    بررسی اعتبار یک درخواست داده

    :param request: درخواست داده به صورت دیکشنری یا DataRequest
    :return: نتیجه اعتبارسنجی
    """
    if isinstance(request, DataRequest):
        req_dict = request.to_dict()
    else:
        req_dict = request

    # بررسی وجود فیلدهای ضروری
    if "metadata" not in req_dict or "payload" not in req_dict:
        return False

    metadata = req_dict["metadata"]
    payload = req_dict["payload"]

    # بررسی فیلدهای ضروری متادیتا
    required_metadata = ["request_id", "timestamp", "source", "destination"]
    if not all(field in metadata for field in required_metadata):
        return False

    # بررسی فیلدهای ضروری payload
    required_payload = ["operation", "model_id", "data_type"]
    if not all(field in payload for field in required_payload):
        return False

    return True


def is_valid_data_response(response: Union[Dict[str, Any], DataResponse]) -> bool:
    """
    بررسی اعتبار یک پاسخ داده

    :param response: پاسخ داده به صورت دیکشنری یا DataResponse
    :return: نتیجه اعتبارسنجی
    """
    if isinstance(response, DataResponse):
        resp_dict = response.to_dict()
    else:
        resp_dict = response

    # بررسی وجود فیلدهای ضروری
    if "metadata" not in resp_dict or "payload" not in resp_dict:
        return False

    metadata = resp_dict["metadata"]
    payload = resp_dict["payload"]

    # بررسی فیلدهای ضروری متادیتا
    required_metadata = ["request_id", "timestamp", "source", "destination"]
    if not all(field in metadata for field in required_metadata):
        return False

    # بررسی فیلدهای ضروری payload
    required_payload = ["operation", "status", "data_type"]
    if not all(field in payload for field in required_payload):
        return False

    return True


