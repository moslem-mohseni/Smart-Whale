"""
file_events.py - Schema for file-related events in ClickHouse

This table stores events specifically related to file operations,
such as uploads, downloads, deletions, and sharing events,
enabling detailed analytics on file usage patterns.
"""

from storage.scripts.base_schema import ClickHouseSchema


class FileEventsTable(ClickHouseSchema):
    """
    Table schema for file-related events
    """

    @property
    def name(self) -> str:
        return "file_events"

    @property
    def description(self) -> str:
        return "Events related to file operations"

    @property
    def database_name(self) -> str:
        return "default"

    def get_create_statement(self) -> str:
        return f"""
        -- Create database if it doesn't exist
        CREATE DATABASE IF NOT EXISTS {self.database_name};

        -- File events table
        CREATE TABLE IF NOT EXISTS {self.database_name}.{self.name} (
            event_date Date,
            event_time DateTime,
            event_type LowCardinality(String), -- 'upload', 'download', 'delete', 'share'
            file_id String,
            file_hash String,
            user_id UInt64,
            file_size UInt64,
            file_type String,
            processing_time_ms UInt32,
            status LowCardinality(String), -- 'success', 'error'
            error_type String,
            ip_address String
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(event_date)
        ORDER BY (event_type, event_time);

        -- Add table comment
        COMMENT ON TABLE {self.database_name}.{self.name} IS 'Events related to file operations';
        """

    def get_check_exists_statement(self) -> str:
        return f"SHOW TABLES FROM {self.database_name} LIKE '{self.name}'"
