"""
usage_stats.py - Schema for usage statistics in ClickHouse

This table stores aggregated usage statistics for models and users,
enabling detailed analytics on system usage patterns and performance.
"""

from storage.scripts.base_schema import ClickHouseSchema


class UsageStatsTable(ClickHouseSchema):
    """
    Table schema for usage statistics
    """

    @property
    def name(self) -> str:
        return "usage_stats"

    @property
    def description(self) -> str:
        return "Aggregated usage statistics for models and users"

    @property
    def database_name(self) -> str:
        return "default"

    def get_create_statement(self) -> str:
        return f"""
        -- Create database if it doesn't exist
        CREATE DATABASE IF NOT EXISTS {self.database_name};

        -- Usage statistics table
        CREATE TABLE IF NOT EXISTS {self.database_name}.{self.name} (
            date Date,
            user_id UInt64,
            model_name LowCardinality(String),
            tokens_input UInt32,
            tokens_output UInt32,
            processing_time_ms UInt32,
            request_count UInt32,
            average_latency_ms Float32,
            error_count UInt16
        ) ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (date, user_id, model_name);
        
        -- Add table comment
        COMMENT ON TABLE {self.database_name}.{self.name} IS 'Aggregated usage statistics for models and users';
        """

    def get_check_exists_statement(self) -> str:
        return f"SHOW TABLES FROM {self.database_name} LIKE '{self.name}'"
