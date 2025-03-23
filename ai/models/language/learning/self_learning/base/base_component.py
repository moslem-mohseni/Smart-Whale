# فایل نهایی BaseComponent با مکانیزم‌های پیشرفته و کامل
# ----------------------------------------------------------------------------

import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union, Callable

from ai.core.logging.handlers.file_handler import FileLogHandler
from ai.core.logging.formatters.json_formatter import JSONLogFormatter
from ai.core.monitoring.metrics.collector import MetricsCollector
from ai.core.cache.manager.cache_manager import CacheManager
from ai.core.resilience.circuit_breaker.breaker_manager import CircuitBreaker
from ai.core.utils.time.time_utils import TimeUtils


class BaseComponent(ABC):
    """
    کلاس پایه برای تمام اجزای سیستم خودآموزی (Self-Learning)

    این کلاس ویژگی‌های مشترک زیر را ارائه می‌دهد:
    1. مدیریت پیکربندی با پشتیبانی از بازنویسی
    2. سیستم لاگینگ با قابلیت‌های پیشرفته (فایل و کنسول)
    3. جمع‌آوری و ارسال متریک‌ها (MetricsCollector)
    4. مدیریت رویدادها با الگوی ناظر (Observer)
    5. ذخیره و بازیابی وضعیت (Persistence)
    6. مکانیزم‌های تحمل خطا و ریکاوری (Circuit Breaker)
    """

    def __init__(
            self,
            component_id: Optional[str] = None,
            component_type: str = "base",
            model_id: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None
    ):
        """
        راه‌اندازی کامپوننت پایه

        Args:
            component_id: شناسه یکتای کامپوننت (اختیاری - در صورت عدم تعیین، یک UUID تولید می‌شود)
            component_type: نوع کامپوننت (به عنوان پیشوند برای لاگینگ و متریک‌ها استفاده می‌شود)
            model_id: شناسه مدل زبانی مرتبط (اختیاری)
            config: پیکربندی اختصاصی (اختیاری)
        """
        # تنظیم شناسه‌های اولیه
        self.component_id = component_id or f"{component_type}-{uuid.uuid4().hex[:8]}"
        self.component_type = component_type
        self.model_id = model_id
        self.start_time = datetime.now()

        # بارگذاری پیکربندی پیش‌فرض و ترکیب با پیکربندی اختصاصی
        self.config = self._load_default_config()
        if config:
            self._merge_config(config)

        # راه‌اندازی لاگینگ پیشرفته
        self.logger = self._setup_logger()

        # راه‌اندازی سیستم متریک
        self.metrics = self._setup_metrics()

        # راه‌اندازی مکانیزم مدیریت رویدادها
        self.event_handlers: Dict[str, Set[Callable]] = {}

        # راه‌اندازی مدیریت وضعیت در حافظه
        self.state: Dict[str, Any] = {}
        self.state_changed = False

        # راه‌اندازی مکانیزم‌های تحمل خطا (Circuit Breaker)
        self.circuit_breaker = self._setup_circuit_breaker()

        # راه‌اندازی کش - پیش‌فرض None (تنظیم در صورت نیاز)
        self.cache = None

        self.logger.info(f"[BaseComponent] {self.component_id} initialized")
        self.increment_metric("component_initialization")

    def _load_default_config(self) -> Dict[str, Any]:
        """
        بارگذاری تنظیمات پیش‌فرض

        Returns:
            Dict[str, Any]: تنظیمات پیش‌فرض
        """
        return {
            "logging": {
                "level": "INFO",
                "format": "default",  # یا "json"
                "file_path": f"logs/{self.component_type}/{self.component_id}.log",
                "max_size": 10 * 1024 * 1024,  # 10MB
                "backup_count": 5
            },
            "metrics": {
                "enabled": True,
                "port": 9100,
                "update_interval": 5
            },
            "state": {
                "persistence": True,
                "auto_save": True,
                "save_interval": 300  # seconds
            },
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "recovery_time": 30
            },
            "cache": {
                "enabled": False  # پیش‌فرض غیرفعال است چون نیاز به redis_client دارد
            },
            "events": {
                "async_processing": True,
                "queue_size": 100
            }
        }

    def _merge_config(self, custom_config: Dict[str, Any]) -> None:
        """
        ترکیب تنظیمات سفارشی با تنظیمات پیش‌فرض

        Args:
            custom_config: تنظیمات سفارشی
        """

        def _recursive_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            """ترکیب بازگشتی دو دیکشنری"""
            for key, value in override.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    _recursive_merge(base[key], value)
                else:
                    base[key] = value
            return base

        _recursive_merge(self.config, custom_config)

    def _setup_logger(self) -> logging.Logger:
        """
        راه‌اندازی سیستم لاگینگ

        Returns:
            logging.Logger: شیء لاگر
        """
        logger_name = f"{self.component_type}.{self.component_id}"
        logger = logging.getLogger(logger_name)

        # تنظیم سطح لاگ
        level_name = self.config["logging"]["level"]
        level = getattr(logging, level_name.upper(), logging.INFO)
        logger.setLevel(level)

        # جلوگیری از تکرار handlers
        if logger.handlers:
            return logger

        # تعیین فرمتر
        if self.config["logging"]["format"] == "json":
            formatter = JSONLogFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        # ساخت فایل لاگ
        file_path = self.config["logging"]["file_path"]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=self.config["logging"]["max_size"],
            backupCount=self.config["logging"]["backup_count"]
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # افزودن console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _setup_metrics(self) -> MetricsCollector:
        """
        راه‌اندازی سیستم متریک

        Returns:
            MetricsCollector: شیء جمع‌کننده متریک‌ها
        """
        if not self.config["metrics"].get("enabled", False):
            return None

        port = self.config["metrics"].get("port", 9100)
        update_interval = self.config["metrics"].get("update_interval", 5)

        return MetricsCollector(port=port, update_interval=update_interval)

    def _setup_circuit_breaker(self) -> CircuitBreaker:
        """
        راه‌اندازی مکانیزم Circuit Breaker

        Returns:
            CircuitBreaker: مدیریت‌کننده Circuit Breaker
        """
        if not self.config["circuit_breaker"].get("enabled", False):
            return None

        threshold = self.config["circuit_breaker"].get("failure_threshold", 5)
        recovery_time = self.config["circuit_breaker"].get("recovery_time", 30)

        return CircuitBreaker(
            failure_threshold=threshold,
            recovery_time=recovery_time
        )

    def setup_cache(self, redis_client: CacheManager) -> None:
        """
        راه‌اندازی سیستم کش

        Args:
            redis_client: سرویس Redis (CacheManager) برای مدیریت کش
        """
        if not self.config["cache"].get("enabled", False):
            return

        self.cache = redis_client
        self.logger.info("Cache system initialized")

    def increment_metric(self, name: str) -> None:
        """
        افزایش یک متریک شمارنده

        Args:
            name: نام متریک
        """
        if not self.metrics:
            return
        try:
            self.metrics.increment_request_count()
            self.logger.debug(f"Incremented metric: {name}")
        except Exception as e:
            self.logger.warning(f"Failed to increment metric {name}: {str(e)}")

    def record_error_metric(self) -> None:
        """
        ثبت یک خطا در متریک‌ها
        """
        if not self.metrics:
            return
        try:
            self.metrics.increment_error_count()
            self.logger.debug("Recorded error metric")
        except Exception as e:
            self.logger.warning(f"Failed to record error metric: {str(e)}")

    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        ثبت یک پردازنده رویداد

        Args:
            event_type: نوع رویداد
            handler: تابع پردازنده رویداد
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = set()
        self.event_handlers[event_type].add(handler)
        self.logger.debug(f"Registered handler for event type: {event_type}")

    def unregister_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        حذف یک پردازنده رویداد

        Args:
            event_type: نوع رویداد
            handler: تابع پردازنده رویداد
        """
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            self.logger.debug(f"Unregistered handler for event type: {event_type}")

    async def trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        ایجاد یک رویداد و ارسال به پردازنده‌های ثبت‌شده

        Args:
            event_type: نوع رویداد
            event_data: داده‌های رویداد
        """
        if event_type not in self.event_handlers:
            self.logger.debug(f"No handlers for event type: {event_type}")
            return

        event_obj = {
            "type": event_type,
            "component_id": self.component_id,
            "timestamp": TimeUtils.get_current_utc(),
            "data": event_data
        }

        self.logger.debug(f"Triggering event: {event_type}")
        self.increment_metric(f"event_{event_type}")

        # بررسی نوع پردازش (همزمان یا ناهمزمان)
        if self.config["events"].get("async_processing", True):
            # پردازش ناهمزمان رویدادها
            tasks = []
            for handler in self.event_handlers[event_type]:
                tasks.append(asyncio.create_task(self._call_event_handler(handler, event_obj)))
            if tasks:
                await asyncio.gather(*tasks)
        else:
            # پردازش همزمان رویدادها
            for handler in self.event_handlers[event_type]:
                await self._call_event_handler(handler, event_obj)

    async def _call_event_handler(self, handler: Callable, event: Dict[str, Any]) -> None:
        """
        فراخوانی یک پردازنده رویداد با مدیریت خطا

        Args:
            handler: تابع پردازنده رویداد
            event: داده‌های رویداد
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            self.logger.error(f"Error in event handler: {str(e)}")
            self.record_error_metric()

    def update_state(self, key: str, value: Any) -> None:
        """
        به‌روزرسانی وضعیت کامپوننت

        Args:
            key: کلید وضعیت
            value: مقدار جدید
        """
        self.state[key] = value
        self.state_changed = True

        # در صورت فعال بودن ذخیره خودکار، وضعیت را ذخیره کن
        if self.config["state"].get("auto_save", True):
            asyncio.create_task(self.save_state())

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        دریافت مقدار وضعیت

        Args:
            key: کلید وضعیت
            default: مقدار پیش‌فرض در صورت عدم وجود کلید

        Returns:
            Any: مقدار وضعیت
        """
        return self.state.get(key, default)

    async def save_state(self) -> bool:
        """
        ذخیره وضعیت کامپوننت

        Returns:
            bool: نتیجه عملیات ذخیره‌سازی
        """
        if not self.config["state"].get("persistence", True):
            return False

        if not self.state_changed:
            return True

        try:
            # ساخت داده‌های وضعیت
            state_data = {
                "component_id": self.component_id,
                "component_type": self.component_type,
                "model_id": self.model_id,
                "timestamp": TimeUtils.get_current_utc(),
                "state": self.state
            }

            # ذخیره‌سازی در فایل
            state_path = f"states/{self.component_type}/{self.component_id}.json"

            # ایجاد دایرکتوری در صورت نیاز
            os.makedirs(os.path.dirname(state_path), exist_ok=True)

            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)

            self.state_changed = False
            self.logger.debug(f"State saved to {state_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")
            self.record_error_metric()
            return False

    async def load_state(self) -> bool:
        """
        بارگذاری وضعیت کامپوننت

        Returns:
            bool: نتیجه عملیات بارگذاری
        """
        if not self.config["state"].get("persistence", True):
            return False

        try:
            state_path = f"states/{self.component_type}/{self.component_id}.json"

            if not os.path.exists(state_path):
                self.logger.debug(f"No saved state found at {state_path}")
                return False

            with open(state_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            if "state" in state_data:
                self.state = state_data["state"]
                self.state_changed = False
                self.logger.info(f"State loaded from {state_path}")
                return True
            else:
                self.logger.warning(f"Invalid state data in {state_path}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to load state: {str(e)}")
            self.record_error_metric()
            return False

    async def get_cache_data(self, key: str) -> Any:
        """
        دریافت مقدار از کش

        Args:
            key: کلید کش

        Returns:
            Any: مقدار کش یا None در صورت عدم وجود
        """
        if not self.cache or not self.config["cache"].get("enabled", False):
            return None

        try:
            return await self.cache.get(key)
        except Exception as e:
            self.logger.debug(f"Cache get error for key {key}: {str(e)}")
            return None

    async def set_cache_data(self, key: str, value: Any) -> bool:
        """
        تنظیم مقدار در کش

        Args:
            key: کلید کش
            value: مقدار

        Returns:
            bool: نتیجه عملیات
        """
        if not self.cache or not self.config["cache"].get("enabled", False):
            return False

        try:
            await self.cache.set(key, value)
            return True
        except Exception as e:
            self.logger.debug(f"Cache set error for key {key}: {str(e)}")
            return False

    async def invalidate_cache(self, key: str) -> bool:
        """
        حذف مقدار از کش

        Args:
            key: کلید کش

        Returns:
            bool: نتیجه عملیات
        """
        if not self.cache or not self.config["cache"].get("enabled", False):
            return False

        try:
            await self.cache.invalidate(key)
            return True
        except Exception as e:
            self.logger.debug(f"Cache invalidation error for key {key}: {str(e)}")
            return False

    async def clear_all_cache(self) -> bool:
        """
        پاکسازی کامل کش

        Returns:
            bool: نتیجه عملیات
        """
        if not self.cache or not self.config["cache"].get("enabled", False):
            return False

        try:
            await self.cache.clear_all()
            return True
        except Exception as e:
            self.logger.debug(f"Cache clear error: {str(e)}")
            return False

    async def execute_with_circuit_breaker(self, func: Callable, *args, **kwargs) -> Any:
        """
        اجرای یک تابع با مکانیزم Circuit Breaker

        Args:
            func: تابع مورد نظر
            *args: آرگومان‌های تابع
            **kwargs: آرگومان‌های کلیدی تابع

        Returns:
            Any: نتیجه اجرای تابع

        Raises:
            Exception: در صورت باز بودن Circuit Breaker یا خطا در اجرا
        """
        if not self.circuit_breaker:
            # اگر Circuit Breaker غیرفعال باشد، تابع را مستقیماً اجرا کن
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        if not self.circuit_breaker.allow_request():
            self.logger.warning(f"Circuit breaker is open, rejecting call to {func.__name__}")
            self.record_error_metric()
            raise Exception(f"Circuit breaker is open for {func.__name__}")

        try:
            # اجرای تابع
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # ثبت اجرای موفق
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            # ثبت خطا
            self.logger.error(f"Error in circuit breaker protected call to {func.__name__}: {str(e)}")
            self.circuit_breaker.record_failure()
            self.record_error_metric()
            raise

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت کامپوننت

        Returns:
            Dict[str, Any]: وضعیت کامپوننت
        """
        uptime = (datetime.now() - self.start_time).total_seconds()

        status = {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "model_id": self.model_id,
            "start_time": self.start_time.isoformat(),
            "uptime_seconds": uptime,
            "state_items": len(self.state),
            "registered_events": {
                event_type: len(handlers)
                for event_type, handlers in self.event_handlers.items()
            },
            "config": {
                "logging_level": self.config["logging"].get("level", "INFO"),
                "metrics_enabled": self.config["metrics"].get("enabled", False),
                "state_persistence": self.config["state"].get("persistence", True),
                "circuit_breaker_enabled": self.config["circuit_breaker"].get("enabled", False),
                "cache_enabled": self.config["cache"].get("enabled", False)
            }
        }

        return status

    @abstractmethod
    async def initialize(self) -> bool:
        """
        مقداردهی اولیه کامپوننت - باید در کلاس‌های فرزند پیاده‌سازی شود

        Returns:
            bool: نتیجه مقداردهی اولیه
        """
        pass

    @abstractmethod
    async def cleanup(self) -> bool:
        """
        پاکسازی منابع کامپوننت - باید در کلاس‌های فرزند پیاده‌سازی شود

        Returns:
            bool: نتیجه پاکسازی
        """
        pass
