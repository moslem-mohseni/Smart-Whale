"""
semantic_usage_stats_table.py - Schema for semantic analysis usage statistics

This table stores performance and usage statistics for semantic analysis operations,
including intent detection, sentiment analysis, and topic categorization.
"""

from storage.scripts.base_schema import ClickHouseSchema


class SemanticUsageStatsTable(ClickHouseSchema):
    """
    Table schema for semantic analysis usage statistics
    """

    @property
    def name(self) -> str:
        return "semantic_usage_stats"

    @property
    def description(self) -> str:
        return "Usage statistics for Persian semantic analysis operations"

    @property
    def database_name(self) -> str:
        return "persian_language"

    def get_create_statement(self) -> str:
        return f"""
        -- Create database if it doesn't exist
        CREATE DATABASE IF NOT EXISTS {self.database_name};

        -- Semantic analysis statistics table
        CREATE TABLE IF NOT EXISTS {self.database_name}.{self.name} (
            date Date,
            operation_type LowCardinality(String), -- 'intent_detection', 'sentiment_analysis', 'topic_categorization', etc.
            intent_type LowCardinality(String) DEFAULT '',
            sentiment LowCardinality(String) DEFAULT '',
            topic_category LowCardinality(String) DEFAULT '',
            request_count UInt32,
            avg_confidence Float32,
            avg_processing_time_ms Float32,
            cache_hits UInt32,
            cache_misses UInt32,
            model_invocations UInt32,
            smart_model_uses UInt32,
            teacher_uses UInt32,
            new_patterns_discovered UInt16,
            errors_count UInt16,
            text_avg_length Float32
        ) ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (date, operation_type)
        SETTINGS index_granularity = 8192;
        
        -- Add table comment
        COMMENT ON TABLE {self.database_name}.{self.name} IS 'This table tracks usage statistics for Persian semantic analysis operations';
        """
