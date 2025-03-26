"""
analysis_metrics_table.py - Schema for analyzer performance metrics

This table stores performance metrics for the Persian language analyzer module,
allowing us to track processing times, cache hit rates, and various performance indicators.
"""

from storage.scripts.base_schema import ClickHouseSchema


class AnalysisMetricsTable(ClickHouseSchema):
    """
    Table schema for analyzer performance metrics
    """

    @property
    def name(self) -> str:
        return "analysis_metrics"

    @property
    def description(self) -> str:
        return "Performance metrics for the Persian language analyzer module"

    @property
    def database_name(self) -> str:
        return "persian_language"

    def get_create_statement(self) -> str:
        return f"""
        -- Create database if it doesn't exist
        CREATE DATABASE IF NOT EXISTS {self.database_name};

        -- Analyzer performance metrics table
        CREATE TABLE IF NOT EXISTS {self.database_name}.{self.name} (
            event_date Date,
            event_time DateTime,
            operation_type LowCardinality(String),
            module_name LowCardinality(String),
            text_length UInt32,
            processing_time_ms Float64,
            cache_hit UInt8,
            model_used LowCardinality(String),
            token_count UInt32,
            confidence Float32,
            error_occurred UInt8,
            error_type LowCardinality(String) DEFAULT '',
            user_id String DEFAULT '',
            session_id String DEFAULT '',
            server_name String,
            system_load Float32
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(event_date)
        ORDER BY (operation_type, event_time)
        SETTINGS index_granularity = 8192;
        
        -- Add table comment
        COMMENT ON TABLE {self.database_name}.{self.name} IS 'This table tracks performance metrics for the Persian language analyzer module';
        """
