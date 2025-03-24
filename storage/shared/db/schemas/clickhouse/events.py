"""
events.py - Schema for system events in ClickHouse

This table stores system-wide events for analytics purposes, capturing
user interactions, system events, and operational metrics.
"""

from storage.scripts.base_schema import ClickHouseSchema


class EventsTable(ClickHouseSchema):
    """
    Table schema for system events
    """

    @property
    def name(self) -> str:
        return "events"

    @property
    def description(self) -> str:
        return "System-wide events for analytics"

    @property
    def database_name(self) -> str:
        return "default"

    def get_create_statement(self) -> str:
        return f"""
        -- Create database if it doesn't exist
        CREATE DATABASE IF NOT EXISTS {self.database_name};

        -- Events table
        CREATE TABLE IF NOT EXISTS {self.database_name}.{self.name} (
            event_date Date,
            event_time DateTime,
            event_type LowCardinality(String),
            user_id UInt64,
            session_id String,
            properties String,  -- JSON string
            ip_address String,
            user_agent String
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(event_date)
        ORDER BY (event_type, event_time);

        -- Add table comment
        COMMENT ON TABLE {self.database_name}.{self.name} IS 'System-wide events for analytics';
        """

    def get_check_exists_statement(self) -> str:
        return f"SHOW TABLES FROM {self.database_name} LIKE '{self.name}'"
    