"""
DefaultConfig Module
-----------------------
این فایل تنظیمات پیش‌فرض ماژول Self-Learning را تعریف می‌کند.
تمامی تنظیمات اولیه و پارامترهای مهم سیستم (مانند نرخ یادگیری، اندازه دسته، آستانه‌های انتقال فاز، تنظیمات لاگینگ و ...) در اینجا تعریف شده‌اند.
این نسخه نهایی و عملیاتی است و می‌تواند به عنوان مرجع اصلی تنظیمات در سراسر سیستم استفاده شود.
"""
from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    "logging": {
        "level": "INFO",
        "format": "default",  # یا "json"
        "file_path": "logs/self_learning.log",
        "max_size": 10 * 1024 * 1024,  # 10 MB
        "backup_count": 5
    },
    "metrics": {
        "enabled": True,
        "port": 9100,
        "update_interval": 10  # ثانیه
    },
    "state": {
        "persistence": True,
        "auto_save": True,
        "save_interval": 300  # ثانیه
    },
    "transition_threshold": 0.1,
    "coverage_threshold": 0.8,
    "training_cycle_interval_seconds": 1800,  # 30 دقیقه
    "learning_rate_adjuster": {
        "initial_lr": 0.01,
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6
    },
    "adaptive_scheduler": {
        "initial_interval": 1800,
        "min_interval": 600,
        "max_interval": 7200,
        "adaptation_factor": 0.1
    },
    "batch_optimizer": {
        "max_batch_size": 32,
        "sort_key": "length",
        "shuffle": False
    },
    "strategy": {
        "beginner": {"learning_rate": 0.01, "batch_size": 16, "priority": "high"},
        "intermediate": {"learning_rate": 0.005, "batch_size": 32, "priority": "medium"},
        "advanced": {"learning_rate": 0.001, "batch_size": 64, "priority": "low"}
    },
    "config": {
        "default_source": "GENERAL",
        "wiki_keywords": ["wikipedia", "wiki"]
    },
    "prometheus": {
        "port": 9090,
        "metrics_path": "/metrics"
    }
}
