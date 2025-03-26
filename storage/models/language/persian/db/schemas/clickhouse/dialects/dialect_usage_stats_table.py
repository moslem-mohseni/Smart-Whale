"""
dialect_usage_stats_table.py - Schema for dialect usage statistics

This table stores statistics about dialect detection and usage, allowing us to track
which dialects are most commonly detected, conversion accuracy, and performance metrics.
"""

from storage.scripts.base_schema import ClickHouseSchema


class DialectUsageStatsTable(ClickHouseSchema):
    """
    Table schema for dialect usage statistics
    """

    @property
    def name(self) -> str:
        return "dialect_usage_stats"

    @property
    def description(self) -> str:
        return "Usage statistics for Persian dialect detection and processing"

    @property
    def database_name(self) -> str:
        return "persian_language"

    def get_create_statement(self) -> str:
        return f"""
        -- Create database if it doesn't exist
        CREATE DATABASE IF NOT EXISTS {self.database_name};

        -- Dialect usage statistics table
        CREATE TABLE IF NOT EXISTS {self.database_name}.{self.name} (
            date Date,
            dialect_id String,
            dialect_code LowCardinality(String),
            dialect_name LowCardinality(String),
            detection_count UInt32,
            conversion_count UInt32,
            text_samples_count UInt32,
            avg_confidence Float32,
            avg_processing_time_ms Float32,
            cache_hits UInt32,
            cache_misses UInt32,
            model_invocations UInt32,
            smart_model_uses UInt32,
            teacher_uses UInt32,
            new_features_discovered UInt16,
            new_words_discovered UInt16,
            errors_count UInt16
        ) ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (date, dialect_id)
        SETTINGS index_granularity = 8192;
        
        -- Add table comment
        COMMENT ON TABLE {self.database_name}.{self.name} IS 'This table tracks usage statistics for Persian dialect processing';
        """
