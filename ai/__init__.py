"""
Smart Whale AI - ماژول هوش مصنوعی

این ماژول دربرگیرنده تمام زیرماژول‌های سیستم هوش مصنوعی Smart Whale است.
"""

# بارگذاری همه زیرماژول‌ها
from . import core
from . import data
from . import models
from . import federation
from . import balance

# تعریف __all__ برای دسترسی به همه‌ی ماژول‌ها با import *
__all__ = [
    "core",
    "data",
    "models",
    "federation",
    "balance"
]

# نسخه ماژول
__version__ = "1.0.0"