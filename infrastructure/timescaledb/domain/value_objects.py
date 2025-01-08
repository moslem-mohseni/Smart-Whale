# infrastructure/timescaledb/domain/value_objects.py
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class TimeRange:
    """مدل مقدار برای نمایش بازه زمانی"""
    start: datetime
    end: datetime

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError("start time must be before end time")