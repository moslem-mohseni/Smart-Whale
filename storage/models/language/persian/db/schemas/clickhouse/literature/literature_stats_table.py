"""
literature_stats_table.py - Schema for literature analysis statistics

This table stores statistics about literary analysis operations, including
literary devices detection, poetry meter analysis, and stylistic analysis.
"""

from storage.scripts.base_schema import ClickHouseSchema


class LiteratureStatsTable(ClickHouseSchema):
    """
    Table schema for literature analysis statistics
    """

    @property
    def name(self) -> str:
        return "literature_stats"

    @property
    def description(self) -> str:
        return "Statistics for Persian literature analysis operations"

    @property
    def database_name(self) -> str:
        return "persian_language"

    def get_create_statement(self) -> str:
        return f"""
        -- Create database if it doesn't exist
        CREATE DATABASE IF NOT EXISTS {self.database_name};

        -- Literature statistics table
        CREATE TABLE IF NOT EXISTS {self.database_name}.{self.name} (
            date Date,
            operation_type LowCardinality(String), -- 'device_detection', 'meter_analysis', 'style_detection', etc.
            literary_type LowCardinality(String),  -- 'poetry', 'prose', etc.
            period_id String,
            style_id String,
            request_count UInt32,
            detection_count UInt32,
            avg_confidence Float32,
            avg_processing_time_ms Float32,
            cache_hits UInt32,
            cache_misses UInt32,
            model_invocations UInt32,
            new_patterns_discovered UInt16,
            corpus_contributions UInt16,
            errors_count UInt16
        ) ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (date, operation_type, literary_type)
        SETTINGS index_granularity = 8192;
        
        -- Add table comment
        COMMENT ON TABLE {self.database_name}.{self.name} IS 'This table tracks statistics for Persian literature analysis';
        """
