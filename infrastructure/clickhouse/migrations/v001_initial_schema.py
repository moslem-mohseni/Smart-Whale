# infrastructure/clickhouse/migrations/v001_initial_schema.py

"""
این فایل اولین مهاجرت برای ایجاد ساختارهای پایه ClickHouse است.
در این مهاجرت، جداول اصلی برای ذخیره‌سازی رویدادها و تحلیل‌ها ایجاد می‌شوند.
همچنین ایندکس‌ها و پارتیشن‌های مناسب برای عملکرد بهینه تعریف می‌شوند.
"""

from typing import Any
from ..adapters.clickhouse_adapter import ClickHouseAdapter


async def upgrade(adapter: ClickHouseAdapter) -> None:
    """
    ایجاد ساختارهای پایه دیتابیس

    این تابع جداول اصلی مورد نیاز برای سیستم تحلیلی را ایجاد می‌کند.
    همچنین تنظیمات لازم برای بهینه‌سازی عملکرد را اعمال می‌نماید.
    """

    # جدول اصلی رویدادها
    await adapter.execute_query("""
        CREATE TABLE IF NOT EXISTS events (
            event_id String,
            event_type LowCardinality(String),
            timestamp DateTime,
            user_id String,
            data String, -- JSON format
            metadata String, -- JSON format

            -- ستون‌های محاسباتی برای تحلیل سریع‌تر
            event_date Date MATERIALIZED toDate(timestamp),
            event_hour UInt8 MATERIALIZED toHour(timestamp)
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (event_type, timestamp, event_id)
        SETTINGS index_granularity = 8192
    """)

    # جدول تجمیعی برای تحلیل‌های روزانه
    await adapter.execute_query("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS daily_event_stats
        ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (date, event_type)
        POPULATE AS
        SELECT
            event_date as date,
            event_type,
            count() as event_count,
            uniqExact(user_id) as unique_users,
            min(timestamp) as first_event,
            max(timestamp) as last_event
        FROM events
        GROUP BY event_date, event_type
    """)

    # ساخت دیکشنری برای event_type ها
    await adapter.execute_query("""
        CREATE DICTIONARY IF NOT EXISTS event_types_dict (
            event_type String,
            category String,
            description String
        )
        PRIMARY KEY event_type
        SOURCE(CLICKHOUSE(TABLE 'event_type_reference'))
        LIFETIME(MIN 0 MAX 3600)
        LAYOUT(FLAT())
    """)

    # جدول مرجع برای event_type ها
    await adapter.execute_query("""
        CREATE TABLE IF NOT EXISTS event_type_reference (
            event_type String,
            category String,
            description String
        )
        ENGINE = MergeTree()
        ORDER BY event_type
    """)


async def downgrade(adapter: ClickHouseAdapter) -> None:
    """
    حذف ساختارهای ایجاد شده

    این تابع در صورت نیاز به بازگشت تغییرات، تمام ساختارهای ایجاد شده را حذف می‌کند.
    """
    # حذف view های تجمیعی
    await adapter.execute_query("DROP VIEW IF EXISTS daily_event_stats")

    # حذف دیکشنری و جدول مرجع
    await adapter.execute_query("DROP DICTIONARY IF EXISTS event_types_dict")
    await adapter.execute_query("DROP TABLE IF EXISTS event_type_reference")

    # حذف جدول اصلی
    await adapter.execute_query("DROP TABLE IF EXISTS events")