"""
Base Package
-------------
این پکیج شامل فایل‌های اصلی و زیربنایی سیستم خودآموزی (Self-Learning) است. ماژول‌های کلیدی آن عبارتند از:
  - base_component: کلاس پایه برای تمام اجزای سیستم
  - config_manager: مدیریت و بارگذاری تنظیمات
  - engine_core: هسته‌ی اصلی موتور یادگیری
  - event_system: سیستم رویدادها و مکانیزم publish/subscribe
  - metrics_collector: جمع‌آوری و گزارش متریک‌های پایه
  - model_interface: رابط ارتباطی با مدل زبانی
  - module_connector: اتصال و هماهنگی با سایر ماژول‌ها (Balance، Data، Federation)
  - pattern_detector: تشخیص الگوهای غیرعادی در داده‌ها
  - phase_definitions: تعریف فازهای تکاملی (BEGINNER، INTERMEDIATE، ADVANCED)
  - phase_detector: تشخیص مرحله فعلی مدل
  - phase_parameter_provider: ارائه پارامترهای متناسب با هر مرحله
  - progress_reporter: گزارش‌دهی پیشرفت مدل در طول یادگیری
  - request_handler: مدیریت درخواست‌های ورودی به صورت ناهمزمان
  - resource_manager: مدیریت منابع پایه (CPU، حافظه، تردها و ...)
  - state_manager: ذخیره و بازیابی وضعیت کلی چرخه‌ی یادگیری
  - training_orchestrator: هماهنگ‌کننده‌ی فرآیند آموزش
  - transition_controller: کنترل‌کننده‌ی گذار بین مراحل تکاملی مدل
  - trend_analyzer: تحلیل روندها و تغییرات مهم در داده‌ها

تمامی فایل‌ها در این پکیج به صورت نهایی و عملیاتی پیاده‌سازی شده‌اند و زیربنای ماژول Self-Learning را تشکیل می‌دهند.
"""

from .base_component import BaseComponent
from .config_manager import ConfigManager
from .engine_core import EngineCore
from .event_system import (
    EventType,
    EventPriority,
    Event,
    EventHandler,
    EventBus,
    EventFilter,
    initialize_event_system,
    shutdown_event_system,
    publish_event,
    subscribe_event,
    unsubscribe_event,
    get_event_system_stats
)
from .metrics_collector import MetricsCollector
from .model_interface import ModelInterface, OperationalModelInterface
from .module_connector import ModuleConnector
from .pattern_detector import PatternDetector
from .phase_definitions import LearningPhase, PhaseDefinition, PhaseDefinitions
from .phase_detector import PhaseDetector
from .phase_parameter_provider import PhaseParameterProvider
from .progress_reporter import ProgressReporter
from .request_handler import RequestHandler
from .resource_manager import ResourceManager
from .state_manager import StateManager
from .training_orchestrator import TrainingOrchestrator
from .transition_controller import TransitionController
from .trend_analyzer import TrendAnalyzer

__all__ = [
    "BaseComponent",
    "ConfigManager",
    "EngineCore",
    "EventType",
    "EventPriority",
    "Event",
    "EventHandler",
    "EventBus",
    "EventFilter",
    "initialize_event_system",
    "shutdown_event_system",
    "publish_event",
    "subscribe_event",
    "unsubscribe_event",
    "get_event_system_stats",
    "MetricsCollector",
    "ModelInterface",
    "OperationalModelInterface",
    "ModuleConnector",
    "PatternDetector",
    "LearningPhase",
    "PhaseDefinition",
    "PhaseDefinitions",
    "PhaseDetector",
    "PhaseParameterProvider",
    "ProgressReporter",
    "RequestHandler",
    "ResourceManager",
    "StateManager",
    "TrainingOrchestrator",
    "TransitionController",
    "TrendAnalyzer"
]
