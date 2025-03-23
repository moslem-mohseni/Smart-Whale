# فایل نهایی event_system.py برای مدیریت رویدادهای سیستم خودآموزی
# ----------------------------------------------------------------------------

"""
سیستم مدیریت رویدادها (Event System) برای ماژول Self-Learning

این ماژول امکان تعریف، انتشار و مدیریت رویدادها را فراهم می‌کند و از مدل
publish/subscribe به‌صورت ناهمزمان بهره می‌برد.
"""

import asyncio
import logging
import uuid
import random
from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Callable, Union, Awaitable, Optional


class EventType(Enum):
    """انواع رویدادهای پایه برای سیستم خودآموزی"""
    MODEL_PHASE_CHANGED = 101
    LEARNING_CYCLE_STARTED = 102
    LEARNING_CYCLE_COMPLETED = 103
    LEARNING_NEED_DETECTED = 201
    DATA_REQUEST_SENT = 202
    DATA_RESPONSE_RECEIVED = 203
    TRAINING_STARTED = 302
    TRAINING_COMPLETED = 303
    EVALUATION_COMPLETED = 402
    RESOURCE_WARNING = 501
    ERROR_OCCURRED = 502
    CONFIG_CHANGED = 503


class EventPriority(Enum):
    """سطوح اولویت رویدادها"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class Event:
    """
    مدل اصلی رویداد در سیستم خودآموزی
    """
    def __init__(
        self,
        event_type: Union[EventType, str],
        data: Dict[str, Any] = None,
        source: str = "system",
        priority: EventPriority = EventPriority.MEDIUM,
        correlation_id: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.type = event_type
        self.data = data or {}
        self.source = source
        self.priority = priority
        self.timestamp = datetime.utcnow()
        self.correlation_id = correlation_id or self.id
        # محاسبه اولویت عددی برای صف
        self.priority_value = self._calculate_priority()

    def _calculate_priority(self) -> float:
        """
        محاسبه‌ی ساده اولویت عددی برای استفاده در صف اولویت‌دار،
        بدون نیاز به بررسی صریح نوع self.priority.
        """
        # اگر self.priority واقعاً از نوع EventPriority باشد، مقدار priority.value برمی‌گردد،
        # در غیراین‌صورت به شکل پیش‌فرض، 3 (معادل MEDIUM).
        priority_num = getattr(self.priority, 'value', 3)
        base_value = float(priority_num)

        noise = random.uniform(-0.05, 0.05)
        return base_value + noise

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل رویداد به دیکشنری"""
        type_name = self.type.name if isinstance(self.type, EventType) else str(self.type)
        priority_name = self.priority.name if isinstance(self.priority, EventPriority) else str(self.priority)
        return {
            "id": self.id,
            "type": type_name,
            "data": self.data,
            "source": self.source,
            "priority": priority_name,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }


# تابع پردازش رویداد
EventHandler = Callable[[Event], Union[None, Awaitable[None]]]


class EventBus:
    """
    گذرگاه رویداد برای ارتباط بین اجزای سیستم خودآموزی
    با استفاده از صف اولویت‌دار برای رویدادها
    """

    def __init__(self, max_queue_size: int = 1000):
        self.logger = logging.getLogger("SelfLearningEventBus")
        self.max_queue_size = max_queue_size
        self.is_processing = False
        self.subscribers: Dict[int, List[tuple]] = {}
        self.event_count = 0
        self.retry_count = 0
        self.failed_count = 0
        self.event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.processing_task = None

    async def start(self):
        if self.is_processing:
            return
        self.is_processing = True
        self.processing_task = asyncio.create_task(self._process_events())
        self.logger.info("[EventBus] Event processing started.")

    async def stop(self):
        if not self.is_processing:
            return
        self.is_processing = False
        if self.processing_task:
            await asyncio.wait_for(self.processing_task, timeout=2.0)
            self.processing_task = None
            self.logger.info("[EventBus] Event processing stopped.")

    async def _process_events(self):
        while self.is_processing:
            try:
                priority_value, event = await self.event_queue.get()
                await self._dispatch_event(event)
                self.event_queue.task_done()
            except Exception as e:
                self.logger.error(f"[EventBus] Error in _process_events: {e}")

            if self.event_queue.empty():
                await asyncio.sleep(0.01)

    async def _dispatch_event(self, event: Event):
        # ارسال رویداد به مشترکین
        for filter_id, subscribers in self.subscribers.items():
            for sub_id, handler, event_filter in subscribers:
                if event_filter.match(event):
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        self.failed_count += 1
                        self.logger.error(f"[EventBus] Error in handler {sub_id}: {str(e)}")

    async def publish(self, event: Event) -> bool:
        """انتشار رویداد در سیستم"""
        try:
            if self.event_queue.qsize() < self.max_queue_size:
                await self.event_queue.put((event.priority_value, event))
                self.event_count += 1
                return True
            else:
                self.logger.warning("[EventBus] Queue is full. Dropping event.")
                return False
        except Exception as e:
            self.logger.error(f"[EventBus] Error publishing event: {str(e)}")
            return False

    async def publish_from_dict(self, event_dict: Dict[str, Any]) -> bool:
        """انتشار رویداد از دیکشنری"""
        try:
            event_type = event_dict.get("type", "UNKNOWN")
            priority_str = event_dict.get("priority", "MEDIUM")
            try:
                if isinstance(event_type, str):
                    event_type = EventType[event_type]
            except KeyError:
                pass
            priority = EventPriority[priority_str] if priority_str in EventPriority.__members__ else EventPriority.MEDIUM

            event = Event(
                event_type=event_type,
                data=event_dict.get("data", {}),
                source=event_dict.get("source", "system"),
                priority=priority,
                correlation_id=event_dict.get("correlation_id")
            )

            return await self.publish(event)
        except Exception as e:
            self.logger.error(f"[EventBus] Error creating event from dict: {str(e)}")
            return False

    def subscribe(self, handler: EventHandler, event_filter: "EventFilter") -> str:
        """اشتراک در رویدادها با استفاده از یک EventFilter"""
        subscriber_id = str(uuid.uuid4())
        filter_id = id(event_filter)
        if filter_id not in self.subscribers:
            self.subscribers[filter_id] = []
        self.subscribers[filter_id].append((subscriber_id, handler, event_filter))
        return subscriber_id

    def unsubscribe(self, subscriber_id: str) -> bool:
        """لغو اشتراک"""
        for filter_id, subscribers in list(self.subscribers.items()):
            for i, (sub_id, handler, event_filter) in enumerate(subscribers):
                if sub_id == subscriber_id:
                    subscribers.pop(i)
                    if not subscribers:
                        del self.subscribers[filter_id]
                    return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار سیستم رویداد"""
        return {
            "queue_size": self.event_queue.qsize(),
            "subscribers_count": sum(len(subs) for subs in self.subscribers.values()),
            "events_processed": self.event_count,
            "retry_count": self.retry_count,
            "failed_count": self.failed_count
        }


class EventFilter:
    """
    فیلتر ساده برای رویدادها (بر اساس نوع رویداد، منبع، حداقل اولویت)
    """
    def __init__(
        self,
        event_types: Optional[List[Union[EventType, str]]] = None,
        sources: Optional[List[str]] = None,
        min_priority: EventPriority = EventPriority.LOW
    ):
        self.event_types = set(event_types) if event_types else None
        self.sources = set(sources) if sources else None
        self.min_priority = min_priority

    def match(self, event: Event) -> bool:
        if self.event_types:
            # بررسی انطباق نوع رویداد
            if isinstance(event.type, EventType):
                if event.type not in self.event_types:
                    return False
            elif event.type not in self.event_types:  # برای str
                return False

        if self.sources and event.source not in self.sources:
            return False

        if isinstance(event.priority, EventPriority):
            if event.priority.value > self.min_priority.value:
                return False
        return True


# نمونه Singleton از EventBus
_event_bus_singleton: Optional[EventBus] = None

async def initialize_event_system():
    global _event_bus_singleton
    if _event_bus_singleton is None:
        _event_bus_singleton = EventBus()
        await _event_bus_singleton.start()

async def shutdown_event_system():
    global _event_bus_singleton
    if _event_bus_singleton:
        await _event_bus_singleton.stop()
        _event_bus_singleton = None

async def publish_event(
    event_type: Union[EventType, str],
    data: Dict[str, Any] = None,
    source: str = "system",
    priority: EventPriority = EventPriority.MEDIUM,
    correlation_id: Optional[str] = None
) -> bool:
    """
    انتشار یک رویداد جدید به ساده‌ترین شکل ممکن
    """
    if not _event_bus_singleton:
        # در صورت عدم راه‌اندازی
        return False

    event = Event(event_type, data or {}, source, priority, correlation_id)
    return await _event_bus_singleton.publish(event)


def subscribe_event(
    handler: EventHandler,
    event_types: Optional[List[Union[EventType, str]]] = None,
    sources: Optional[List[str]] = None,
    min_priority: EventPriority = EventPriority.LOW
) -> str:
    """
    اشتراک ساده رویداد
    """
    if not _event_bus_singleton:
        return ""
    event_filter = EventFilter(event_types, sources, min_priority)
    return _event_bus_singleton.subscribe(handler, event_filter)


def unsubscribe_event(subscriber_id: str) -> bool:
    if not _event_bus_singleton:
        return False
    return _event_bus_singleton.unsubscribe(subscriber_id)


def get_event_system_stats() -> Dict[str, Any]:
    if not _event_bus_singleton:
        return {}
    return _event_bus_singleton.get_stats()
